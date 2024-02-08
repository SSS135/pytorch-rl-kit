from gymnasium import register
from procgen.env import ENV_NAMES


def register_environments():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-easy-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name, "distribution_mode": "easy"},
        )