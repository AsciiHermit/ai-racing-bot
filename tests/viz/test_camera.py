from ars.viz.camera import Camera


def test_world_origin_maps_to_screen_center():
    cam = Camera(screen_width=800, screen_height=600, pixels_per_meter=5.0)
    sx, sy = cam.world_to_screen(0.0, 0.0)
    assert sx == 400
    assert sy == 300


def test_positive_world_y_maps_up_on_screen():
    cam = Camera(screen_width=800, screen_height=600, pixels_per_meter=1.0)
    _, sy_origin = cam.world_to_screen(0.0, 0.0)
    _, sy_up = cam.world_to_screen(0.0, 10.0)
    assert sy_up < sy_origin


def test_fit_to_bounds_centers_camera():
    cam = Camera(screen_width=800, screen_height=600)
    cam.fit_to_bounds(min_x=0.0, max_x=100.0, min_y=0.0, max_y=50.0)
    assert cam.center_x == 50.0
    assert cam.center_y == 25.0


def test_fit_to_bounds_keeps_content_on_screen():
    cam = Camera(screen_width=800, screen_height=600)
    cam.fit_to_bounds(min_x=-20.0, max_x=20.0, min_y=-10.0, max_y=10.0)
    for wx, wy in [(-20.0, -10.0), (20.0, 10.0), (0.0, 0.0)]:
        sx, sy = cam.world_to_screen(wx, wy)
        assert 0 <= sx <= 800
        assert 0 <= sy <= 600
