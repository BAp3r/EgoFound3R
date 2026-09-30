from __future__ import annotations


import torch


CONTACT_LOG_CONF_MIN = 0.0


CONTACT_LOG_CONF_MAX = 2.0


def contact_confidence_from_log(log_conf: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_conf.clamp(CONTACT_LOG_CONF_MIN, CONTACT_LOG_CONF_MAX))
