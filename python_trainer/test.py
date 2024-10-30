import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from data_channel import DataChannel
from environment_parameters import set_parameters
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.timers import time
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model")
parser.add_argument(
    "-e",
    "--env",
    default=str(Path(__file__).parent / "build" / "StandaloneLinux64" / "selfDriving"),
)
args = parser.parse_args()

TESTED_MODEL_PATH = args.model
ENV_PATH = args.env


def print_env_info(env: UnityEnvironment) -> None:
    # get the action space and observation space
    print(env.behavior_specs)  # noqa: T201
    behavior_name = next(iter(env.behavior_specs))
    print(f"Name of the behavior : {behavior_name}")  # noqa: T201
    spec = env.behavior_specs[behavior_name]
    observation_shape = spec.observation_specs
    print(observation_shape)  # noqa: T201
    num_actions = spec.action_spec
    print(num_actions)  # noqa: T201


def test_model(model, env) -> None:
    behavior_name = next(iter(env.behavior_specs))
    env.reset()

    just_spawned = 3

    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        dis_action_values = []
        cont_action_values = []

        if decision_steps:
            for agent_id in decision_steps:
                state_obs, _ = Trainer.get_state_and_reward(decision_steps[agent_id])
                _, action = model.run(
                    None,
                    {
                        "vis_obs": np.expand_dims(state_obs[0], axis=0),
                        "nonvis_obs": np.expand_dims(state_obs[1], axis=0),
                    },
                )

                dis_action_values.append(action[0])
                cont_action_values.append([])

            action_tuple = ActionTuple()
            final_dis_action_values = np.array(dis_action_values)
            final_cont_action_values = np.array(cont_action_values)
            action_tuple.add_discrete(final_dis_action_values)
            action_tuple.add_continuous(final_cont_action_values)
            env.set_actions(behavior_name, action_tuple)

        elif len(terminal_steps) != 0:
            if not just_spawned:
                break
            just_spawned -= 1

        env.step()


if __name__ == "__main__":
    # Set up the environment
    engine_channel = EngineConfigurationChannel()
    data_channel = DataChannel()
    env_location = ENV_PATH
    env = UnityEnvironment(
        file_name=env_location,
        num_areas=1,
        side_channels=[engine_channel, data_channel],
    )

    set_parameters(data_channel)

    engine_channel.set_configuration_parameters(time_scale=1)
    env.reset()

    print_env_info(env)

    model = ort.InferenceSession(TESTED_MODEL_PATH)

    time_begin = time.time()
    try:
        test_model(model, env)

    except KeyboardInterrupt:
        print(  # noqa: T201
            "\nTraining interrupted, continue to next cell to save to save the model.",
        )

    finally:
        time_end = time.time()
        print(f"Drove for: {time_end - time_begin}")
        env.close()
