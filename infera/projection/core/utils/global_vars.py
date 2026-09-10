###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from types import SimpleNamespace

from infera.projection.core.launcher.config import InferaSimConfig

_GLOBAL_CLI_ARGS = None
_GLOBAL_CFG = None
_GLOBAL_TARGET_PLATFORM = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def is_initialized():
    if _GLOBAL_CFG is None:
        return False
    return _GLOBAL_CFG.initialized


def set_initialized():
    _GLOBAL_CFG.initialized = True


def get_cli_args():
    """Return cli arguments."""
    _ensure_var_is_initialized(_GLOBAL_CLI_ARGS, "cli args")
    return _GLOBAL_CFG.cli_args


def get_config():
    """Return the global config."""
    _ensure_var_is_initialized(_GLOBAL_CFG, "config")
    return _GLOBAL_CFG


def get_target_platform():
    """Return target platform."""
    _ensure_var_is_initialized(_GLOBAL_TARGET_PLATFORM, "target_platform")
    return _GLOBAL_TARGET_PLATFORM


def set_global_variables(cfg: InferaSimConfig):
    """Set global vars"""
    assert cfg is not None

    global _GLOBAL_CFG
    if _GLOBAL_CFG:
        return
    _GLOBAL_CFG = cfg

    _set_cli_args(cfg)
    _set_target_platform(cfg)


def _set_cli_args(cfg: InferaSimConfig):
    global _GLOBAL_CLI_ARGS
    if _GLOBAL_CLI_ARGS:
        return
    _GLOBAL_CLI_ARGS = cfg.cli_args


def _set_target_platform(cfg: InferaSimConfig):
    global _GLOBAL_TARGET_PLATFORM
    if _GLOBAL_TARGET_PLATFORM:
        return

    platform_config = getattr(cfg, "platform_config", None)
    if platform_config is None:
        # Fallback for configs that don't provide a platform/platform_config.
        # Defaults to a simple local platform with INFO sink level.
        platform_config = SimpleNamespace(
            name="local",
            master_sink_level="INFO",
            gpus_per_node_env_key=None,
        )
        setattr(cfg, "platform_config", platform_config)
    if platform_config.name and platform_config.name != "local":
        from infera.projection.platforms import RemotePlatform

        _GLOBAL_TARGET_PLATFORM = RemotePlatform(platform_config.name)
    else:
        from infera.projection.platforms import LocalPlatform

        _GLOBAL_TARGET_PLATFORM = LocalPlatform("local")
