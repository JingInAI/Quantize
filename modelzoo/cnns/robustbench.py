"""
RobustBench
version: 0.0.1
update: 2024-05-16
"""
try:
    from robustbench import load_model
except ImportError:
    raise ImportError(
        "Please install robustbench to use this model: pip install git+https://github.com/RobustBench/robustbench.git")

from modelzoo.load import MODELS

MODELS.register({
    'rb_wrn-28-10': lambda **kwargs : load_model(model_name='Standard', dataset='cifar10', threat_model='Linf'),
})
