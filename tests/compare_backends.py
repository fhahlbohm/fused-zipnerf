"""Compare native kernels against an upstream checkout without installing either.

Run in a CUDA development environment with PyTorch and a C++ compiler:
    python tests/compare_backends.py --baseline ../fused-zipnerf-upstream
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def load_backend(root, directory, name, fixed_heuristic=True):
    backend = root / 'ZipNeRFCudaBackend' / 'ZipNeRFCudaBackend'
    directory.mkdir(parents=True, exist_ok=True)
    # Both libraries bind the same C++ types. Keep their Python registrations local.
    source = (backend / 'torch_bindings' / 'bindings.cu').read_text()
    for cls in ['Featurizer', 'NerfMlp', 'Adam']:
        source = source.replace(f'(m, "{cls}")', f'(m, "{cls}", py::module_local())')
    bindings = directory / 'bindings.cu'
    if not bindings.exists() or bindings.read_text() != source:
        bindings.write_text(source)
    sources = sorted(backend.glob('*/src/**/*.cu'))
    if fixed_heuristic:
        # Independent timing-based autotuning can select different reductions.
        # Pin both numerical controls to the first cuBLASLt heuristic; production
        # sources and normal performance measurements retain upstream autotuning.
        mlp = backend / 'mlp/src/nerf_mlp.cu'
        source = mlp.read_text().replace('constexpr int max_candidates = 8;', 'constexpr int max_candidates = 1;')
        controlled_mlp = directory / 'nerf_mlp.cu'
        if not controlled_mlp.exists() or controlled_mlp.read_text() != source:
            controlled_mlp.write_text(source)
        sources[sources.index(mlp)] = controlled_mlp
    cflags = ['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17']
    libraries = ['cuda.lib', 'nvrtc.lib', 'cublasLt.lib'] if os.name == 'nt' else ['-lcuda', '-lnvrtc', '-lcublasLt']
    extension = load(
        name=name,
        sources=[str(bindings), *map(str, sources)],
        extra_include_paths=list(map(str, sorted(backend.glob('*/include')))),
        extra_cflags=cflags,
        extra_cuda_cflags=['-std=c++17', '--expt-relaxed-constexpr', '-O3', '--use_fast_math'],
        extra_ldflags=libraries,
        build_directory=str(directory),
    )
    extension.set_rtc_dirs(str(directory / 'rtc'), str(backend / 'featurization' / 'rtc'))
    return extension


def difference(actual, reference):
    a, b = actual.double(), reference.double()
    return {
        'exact': torch.equal(actual, reference),
        'max_abs': (a - b).abs().max().item() if a.numel() else 0.0,
        'relative_l2': ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
    }


def check_gradients(actual, reference, repeat):
    result, noise = difference(actual, reference), difference(repeat, reference)
    assert torch.isfinite(actual).all().item()
    # FP16 atomics depend on warp arrival order. Preserve upstream's mixed split;
    # allow four FP16 epsilons, or the measured repeated-reference noise if larger.
    tolerance = 4 * torch.finfo(torch.float16).eps if actual.dtype == torch.float16 else 1e-5
    scale = reference.double().abs().max().item() if reference.numel() else 0.0
    absolute = max(2e-7 + tolerance * scale, 10 * noise['max_abs'])
    relative = max(tolerance, 10 * noise['relative_l2'])
    assert result['max_abs'] <= absolute and result['relative_l2'] <= relative, result
    return result | {'absolute_limit': absolute, 'relative_limit': relative, 'reference_noise': noise}


def check_grids(extensions):
    result = []
    empty = torch.empty(0, device='cuda')
    for features in [1, 2, 3, 4]:
        sizes, entries = [2, 4, 128], [8, 64, 1 << 20]
        table = torch.randn(sum(entries), features, device='cuda') * 0.01
        magnitudes = torch.tensor([0.03, 0.05, 0.07], device='cuda')
        for rays, intervals, seed in [(1, 1, -1), (33, 5, 37), (65, 3, -1)]:
            origins = torch.randn(rays, 3, device='cuda') * 3
            origins[0] = torch.tensor([-10000.0, 0, 0], device='cuda')
            directions = torch.nn.functional.normalize(torch.randn_like(origins), dim=1)
            radii = torch.rand(rays, device='cuda') * 0.01
            tdist = torch.linspace(0.1, 2.5, intervals + 1, device='cuda')[None].repeat(rays, 1)
            count, width = rays * intervals, len(sizes) * (features + 1)
            gradient = (torch.randn(count, width + 3, device='cuda') * 1e-5).bfloat16()
            encoded = torch.randn(rays, 32, device='cuda', dtype=torch.bfloat16)
            saved = {}
            for name, extension in extensions.items():
                f = extension.Featurizer(2, sizes, entries, features, 0)
                outputs = []
                # Float output, packed BF16 output, and a deliberately unaligned fallback.
                for layout in ['float', 'packed', 'unaligned']:
                    padded = (width + 7) // 8 * 8
                    storage = torch.empty(count * padded + (layout == 'unaligned'), device='cuda', dtype=torch.bfloat16)
                    bf16 = storage[int(layout == 'unaligned'):].reshape(count, padded)
                    floats = torch.empty(count, width, device='cuda') if layout == 'float' else empty
                    inputs = torch.zeros(count, 96, device='cuda', dtype=torch.bfloat16)
                    f.forward(origins, directions, radii, tdist, seed, True, table, magnitudes,
                              empty, empty, empty, empty, empty, floats, bf16, encoded, inputs)
                    assert not bf16[:, width:].count_nonzero().item()
                    assert (inputs[:, -1] == 1).all().item()
                    outputs.extend([floats.clone(), bf16.clone(), inputs.clone()])
                split = f.n_fp32_grad_rows
                gradients = []
                for _ in range(2 if name == 'baseline' else 1):
                    g32 = torch.zeros_like(table[:split])
                    g16 = torch.zeros_like(table[split:], dtype=torch.float16)
                    f.backward(origins, directions, radii, tdist, seed, gradient[:, :width], g32, g16)
                    gradients.append([g32.clone(), g16.clone()])
                saved[name] = outputs, gradients
            for a, b in zip(saved['candidate'][0], saved['baseline'][0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            grads = [check_gradients(a, b, r) for a, b, r in zip(
                saved['candidate'][1][0], saved['baseline'][1][0], saved['baseline'][1][1])]
            result.append({'features': features, 'rays': rays, 'intervals': intervals, 'seed': seed,
                           'float_packed_unaligned_exact': True, 'gradients': grads})
    return result


def check_proposals(extensions):
    result = []
    empty = torch.empty(0, device='cuda')
    for levels in [6, 8]:
        sizes, entries = [2 ** (i + 1) for i in range(levels)], [min(2 ** (3 * (i + 1)), 4096) for i in range(levels)]
        table = torch.randn(sum(entries), 1, device='cuda') * 0.01
        scales = torch.full((levels,), 0.05, device='cuda')
        width = levels * 2
        w0, b0 = torch.randn(64, width, device='cuda') * 0.01, torch.randn(64, device='cuda') * 0.01
        w1, b1 = torch.randn(1, 64, device='cuda') * 0.01, torch.randn(1, device='cuda') * 0.01
        for rays, intervals in [(1, 1), (33, 5), (65, 3), (129, 32)]:
            origins = torch.randn(rays, 3, device='cuda') * 3
            directions = torch.nn.functional.normalize(torch.randn_like(origins), dim=1)
            radii = torch.rand(rays, device='cuda') * 0.01
            tdist = torch.linspace(0.1, 3, intervals + 1, device='cuda')[None].repeat(rays, 1)
            for seed in [-1, 37]:
                saved = []
                for extension in extensions.values():
                    f = extension.Featurizer(0, sizes, entries, 1, 64)
                    for interleave in [False, True]:
                        density = torch.empty(rays, intervals, device='cuda')
                        features = torch.empty(rays * intervals, width, device='cuda')
                        f.forward(origins, directions, radii, tdist, seed, interleave, table, scales,
                                  w0, b0, w1, b1, density, features, empty, empty, empty)
                        saved.append((density, features))
                for pair in saved[1:]:
                    for a, b in zip(pair, saved[0]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                result.append({'levels': levels, 'rays': rays, 'intervals': intervals, 'seed': seed, 'exact': True})
    return result


def check_mlps(extensions):
    result = []
    for density, width, degree in [(64, 256, 4), (32, 256, 4), (64, 128, 4), (64, 256, 3)]:
        shapes = extensions['baseline'].nerf_mlp_parameters(50, density, 256, width, degree)
        # Dyadic inputs limit reduction-order differences between independently
        # autotuned cuBLASLt handles while exercising signs, padding and ReLU.
        params = torch.randint(-2, 3, (sum(math.prod(shape) for _, shape, _ in shapes),), device='cuda').float() / 16
        models = {n: e.NerfMlp(50, density, 256, width, degree) for n, e in extensions.items()}
        packed = {n: m.pack(params) for n, m in models.items()}
        torch.testing.assert_close(packed['candidate'], packed['baseline'], rtol=0, atol=0)
        for rows in [0, 1, 63, 64, 65, 128, 192, 511, 512]:
            features = (torch.randint(-2, 3, (rows, 56), device='cuda').float() / 16).bfloat16()
            inputs = (torch.randint(-2, 3, (rows, models['baseline'].n_inputs), device='cuda').float() / 16).bfloat16()
            inputs[:, -1] = 1
            saved = {n: m.forward(features, inputs.clone(), packed[n], False)[0] for n, m in models.items()}
            torch.testing.assert_close(saved['candidate'], saved['baseline'], rtol=0, atol=0)
        # Training must retain the original GEMM forward/backward path.
        features = (torch.randint(-2, 3, (64, 56), device='cuda').float() / 16).bfloat16()
        inputs = (torch.randint(-2, 3, (64, models['baseline'].n_inputs), device='cuda').float() / 16).bfloat16()
        inputs[:, -1] = 1
        upstream_gradient = (torch.randint(-2, 3, (64, 8), device='cuda').float() / 256).bfloat16()
        saved = {}
        for name, model in models.items():
            output, context = model.forward(features, inputs.clone(), packed[name], True)
            gparams = torch.zeros_like(params)
            gfeatures = model.backward(upstream_gradient, features, packed[name], context, params, gparams)
            saved[name] = [output, gfeatures, gparams]
        for a, b in zip(saved['candidate'], saved['baseline']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        result.append({'density_width': density, 'view_width': width, 'degree': degree,
                       'inference_and_tails_exact': True, 'training_forward_backward_exact': True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--build-directory', type=Path, default=Path('.cache/compare_backends'))
    parser.add_argument('--output', type=Path, default=Path('.cache/compare_backends/results.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(87)
    extensions = {n: load_backend(r.resolve(), (args.build_directory / n).resolve(), f'zipnerf_compare_{n}')
                  for n, r in [('baseline', args.baseline), ('candidate', root)]}
    with torch.no_grad():
        result = {'device': torch.cuda.get_device_name(),
                  'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((root / 'ZipNeRFCudaBackend').rglob('*')) if p.suffix in ['.cu', '.cuh', '.h']},
                  'grids': check_grids(extensions), 'proposals': check_proposals(extensions), 'mlps': check_mlps(extensions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Passed grid, mixed-gradient, proposal and MLP comparisons; results: {args.output}')


if __name__ == '__main__':
    main()
