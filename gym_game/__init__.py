from gym.envs.registration import register

register(
    id='frankaEmikaGame-v0',
    entry_point='gym_game.envs:CatchMeIfYouCanEnv',
)