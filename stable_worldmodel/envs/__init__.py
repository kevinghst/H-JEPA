from gymnasium.envs import registration


WORLDS = set()


def register(id, entry_point):
    registration.register(id=id, entry_point=entry_point)
    WORLDS.add(id)


register(
    id='swm/PushT-v1',
    entry_point='stable_worldmodel.envs.pusht.env:PushT',
)

register(
    id='swm/FourRoomDistractors-v0',
    entry_point='stable_worldmodel.envs.four_room.env:FourRoomDistractorsEnv',
)

register(
    id='swm/OGBCube-v0',
    entry_point='stable_worldmodel.envs.ogbench.cube_env:CubeEnv',
)

register(
    id='swm/OGBAntMaze-v0',
    entry_point='stable_worldmodel.envs.ogbench.antmaze_env:AntMazeEnv',
)
