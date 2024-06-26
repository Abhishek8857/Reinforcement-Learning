from gymnasium.envs.registration import register

register(
    id='CustomFrozenLake-v1',
    entry_point='custom_envs.envs.frozen_lake:FrozenLakeEnv',
)
register(
    id='RecyclingRobot-v1',
    entry_point='custom_envs.envs.recycling_robot:RecyclingRobotEnv',
)
register(
    id='BlackJack-v1',
    entry_point='custom_envs.envs.black_jack:BlackjackEnv',
)
register(
    id='CliffWalking-v1',
    entry_point='custom_envs.envs.cliff_walking:CliffWalkingEnv',
)
register(
    id='CustomCartPole-v1',
    entry_point='custom_envs.envs.cart_pole:CartPoleEnv',
)
register(
    id='CustomCartPole-v2',
    entry_point='custom_envs.envs.cart_pole_v2:CartPoleEnv_v2',
)
register(
    id='CustomPendulum-v1',
    entry_point='custom_envs.envs.pendulum:CustomPendulumEnv',
    max_episode_steps=200,
)