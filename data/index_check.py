import carla

# Carla 클라이언트 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10)

# 월드 및 맵 가져오기
world = client.get_world()
map = world.get_map()

# 스폰 포인트 리스트
spawn_points = map.get_spawn_points()

# 디버그용 텍스트 그리기
for idx, spawn_point in enumerate(spawn_points):
    location = spawn_point.location + carla.Location(z=3.0)  # 지면에서 살짝 띄움
    world.debug.draw_string(
        location,
        text=f'{idx}',
        draw_shadow=False,
        color=carla.Color(r=255, g=0, b=0),
        life_time=60, 		# 60 초
        persistent_lines=True
    )
