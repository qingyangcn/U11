from config.default import AppConfig, build_default_config


def build_config() -> AppConfig:
    cfg = build_default_config()
    cfg.env.load_scale = 1.0
    return cfg
