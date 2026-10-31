from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .ZipNeRFCudaBackend.torch_bindings.pipeline import grid_max_value, nerf_grid_grad_scale, Featurizer, nerf_mlp_parameters, NerfMlp, Adam, \
        generate_rays, view_rays, initialize_rays, sample_uniforms, composite_and_resample, train_final, composite_final
    __all__ = [
        'grid_max_value', 'nerf_grid_grad_scale', 'Featurizer', 'nerf_mlp_parameters', 'NerfMlp', 'Adam', 'generate_rays', 'view_rays', 'initialize_rays',
        'sample_uniforms', 'composite_and_resample', 'train_final', 'composite_final',
    ]
    # set up runtime compilation (the featurization kernels are read from the source tree, compiled kernels are cached)
    from ZipNeRFCudaBackend import _C
    _C.set_rtc_dirs(str(extension_dir / 'rtc' / 'cache'), str(extension_dir / 'ZipNeRFCudaBackend' / 'featurization' / 'rtc'))
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
