"""Regression checks for deterministic inference; run after installing the CUDA backend.

    python -m unittest discover -s tests -v
"""

import importlib
import math
import unittest

import torch


class InferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is required')
        cls.backend = importlib.import_module('ZipNeRFCudaBackend._C')
        # power-ladder warp in the public API's (near, far, fn_near, fn_far, p, premult, y_max) order
        near, far, power, premult = 0.1, 10.0, -1.5, 2.0
        def fn(x: float) -> float:
            return abs(power - 1) / power * ((x * premult / abs(power - 1) + 1) ** power - 1)
        cls.warp = (near, far, fn(near), fn(far), power, premult, 1.6666665)

    def setUp(self) -> None:
        torch.manual_seed(0)

    def assert_exact(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all().item())

    def make_mlp(self, density: int = 64, width: int = 128, degree: int = 4):
        shapes = self.backend.nerf_mlp_parameters(50, density, 64, width, degree)
        params = torch.randn(sum(math.prod(shape) for _, shape, _ in shapes), device='cuda') * 0.02
        mlp = self.backend.NerfMlp(50, density, 64, width, degree)
        return mlp, mlp.pack(params)

    def test_view_rays(self) -> None:
        local = torch.randn(83, 3, device='cuda')
        c2w = torch.randn(3, 4, device='cuda')
        for rays in [0, 1, 7, 32, 65]:
            pixels = torch.randperm(83, device='cuda')[:rays]
            for degree in [0, 1, 4, 5]:
                for samples in [2, 32, 64]:
                    with self.subTest(rays=rays, degree=degree, samples=samples):
                        _, sdist, tdist = self.backend.initialize_rays(-1, local[:1], degree, samples, self.warp)
                        actual = self.backend.view_rays_cached(pixels, local, c2w, 0.01, degree, sdist, tdist)
                        expected = self.backend.view_rays(pixels, local, c2w, 0.01, degree, samples, self.warp)
                        for a, b in zip(actual, expected, strict=True):
                            self.assert_exact(a, b)

    def test_resampling(self) -> None:
        for rays in [0, 1, 7, 32, 65]:
            directions = torch.randn(rays, 3, device='cuda')
            for intervals, samples in [(2, 32), (3, 32), (32, 64), (64, 32), (64, 64)]:
                for case in ['random', 'empty', 'opaque', 'duplicate', 'all_duplicate']:
                    with self.subTest(rays=rays, intervals=intervals, samples=samples, case=case):
                        raw = torch.randn(rays, intervals, device='cuda') * 2
                        sdist = torch.linspace(0, 1, intervals + 1, device='cuda')[None].repeat(rays, 1)
                        tdist = sdist * 10 + 0.1
                        if case == 'empty':
                            raw.fill_(-100)
                        elif case == 'opaque':
                            raw.fill_(20)
                        elif case == 'duplicate':
                            sdist[:, 1] = sdist[:, 0]
                            tdist[:, 1] = tdist[:, 0]
                        elif case == 'all_duplicate':
                            sdist.zero_()
                            tdist.fill_(0.1)
                        _, next_s, next_t = self.backend.composite_and_resample(
                            raw, sdist, tdist, directions, 1.0, samples, -1, 1, self.warp)
                        actual = self.backend.resample_inference(raw, sdist, tdist, directions, samples, self.warp)
                        for a, b in zip(actual, [next_s, next_t], strict=True):
                            self.assert_exact(a, b)

    def test_per_ray_mlp(self) -> None:
        # both fused tile sizes, empty/partial batches, and other layouts' original inference path
        for density, width, degree in [(64, 128, 4), (64, 256, 4), (32, 192, 3)]:
            mlp, packed = self.make_mlp(density, width, degree)
            encoded_width = mlp.n_inputs - density
            for rays, intervals in [(0, 32), (1, 1), (1, 32), (2, 32), (4, 32), (6, 32), (65, 3), (1, 128)]:
                with self.subTest(density=density, width=width, degree=degree, rays=rays, intervals=intervals):
                    features = (torch.randn(rays * intervals, mlp.n_features, device='cuda') * 0.1).bfloat16()
                    encoded = torch.randn(rays, encoded_width, device='cuda').bfloat16()
                    inputs = torch.zeros(rays * intervals, mlp.n_inputs, device='cuda', dtype=torch.bfloat16)
                    inputs[:, -encoded_width:] = encoded.repeat_interleave(intervals, 0)
                    inputs[:, -1] = 1
                    expected, _ = mlp.forward(features, inputs, packed, False)
                    self.assert_exact(mlp.forward_rays(features, encoded, packed, intervals), expected)

    def test_nondefault_streams(self) -> None:
        mlp, packed = self.make_mlp()
        features = torch.randn(256, mlp.n_features, device='cuda').bfloat16()
        encoded = torch.randn(8, 32, device='cuda').bfloat16()
        expected = mlp.forward_rays(features, encoded, packed, 32)
        caller = torch.cuda.current_stream()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for stream in streams:
            stream.wait_stream(caller)
        outputs = []
        for index in range(16):
            with torch.cuda.stream(streams[index % 2]):
                outputs.append(mlp.forward_rays(features, encoded, packed, 32))
        for stream in streams:
            caller.wait_stream(stream)
        for output in outputs:
            self.assert_exact(output, expected)

    def test_invalid_shapes(self) -> None:
        mlp, packed = self.make_mlp()
        features = torch.zeros(128, mlp.n_features, device='cuda', dtype=torch.bfloat16)
        encoded = torch.zeros(4, 32, device='cuda', dtype=torch.bfloat16)
        for intervals in [0, -1, 31]:
            with self.subTest(intervals=intervals), self.assertRaises(RuntimeError):
                mlp.forward_rays(features, encoded, packed, intervals)
        local = torch.randn(3, 3, device='cuda')
        pixels = torch.arange(3, device='cuda')
        c2w = torch.eye(4, device='cuda')[:3].contiguous()
        sdist = torch.zeros(2, 65, device='cuda')
        with self.assertRaises(RuntimeError):
            self.backend.view_rays_cached(pixels, local, c2w, 0.01, 4, sdist, sdist)
        raw = torch.zeros(3, 64, device='cuda')
        for samples in [1, 65]:
            with self.subTest(samples=samples), self.assertRaises(RuntimeError):
                self.backend.resample_inference(raw, sdist, sdist, local, samples, self.warp)


if __name__ == '__main__':
    unittest.main()
