from ars.env.curriculum import RegretCurriculumTrackProvider
from ars.track import make_simple_oval


def _pool(n=4):
    return [make_simple_oval(turn_radius=15.0 + 5.0 * i) for i in range(n)]


def test_sampling_is_deterministic_for_a_given_seed():
    pool = _pool()
    a = RegretCurriculumTrackProvider(pool, seed=0)
    b = RegretCurriculumTrackProvider(pool, seed=0)
    picks_a = [a() for _ in range(10)]
    picks_b = [b() for _ in range(10)]
    assert [t.length for t in picks_a] == [t.length for t in picks_b]


def test_record_episode_applies_to_the_most_recently_sampled_track():
    pool = _pool()
    provider = RegretCurriculumTrackProvider(pool, refresh_every=1, seed=0)
    track = provider()
    idx = pool.index(track)
    provider.record_episode(42.0)
    assert provider.rolling_best[idx] == 42.0


def test_record_episode_with_explicit_track_index_ignores_the_last_sampled_track():
    # With several vectorized envs sharing one provider, the most recently
    # sampled track belongs to whichever env reset last, not necessarily the
    # env whose episode just finished -- callers must be able to say which.
    pool = _pool()
    provider = RegretCurriculumTrackProvider(pool, refresh_every=100, seed=0)
    provider()
    provider()  # some other env's reset moves _last_index
    provider.record_episode(7.0, track_index=2)
    assert provider.rolling_best[2] == 7.0
    assert all(provider.rolling_best[i] == float("-inf") for i in (0, 1, 3))


def test_index_of_returns_the_pool_index_and_rejects_unknown_tracks():
    pool = _pool()
    provider = RegretCurriculumTrackProvider(pool, seed=0)
    assert provider.index_of(pool[2]) == 2
    try:
        provider.index_of(make_simple_oval(turn_radius=99.0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a track outside the pool")


def test_weights_refresh_after_refresh_every_episodes_and_favor_high_regret_tracks():
    pool = _pool(n=2)
    provider = RegretCurriculumTrackProvider(pool, refresh_every=2, seed=0)

    # Manually drive two full (sample, record) cycles per track so both
    # get a rolling best, then create an artificial regret gap: track 0
    # is currently far below its own best (high regret), track 1 is
    # right at its own best (zero regret).
    for track, first_return, second_return in [(pool[0], 100.0, 10.0), (pool[1], 100.0, 100.0)]:
        idx = pool.index(track)
        provider._rolling_best[idx] = first_return
        provider._last_return[idx] = second_return
        provider._visited[idx] = True

    provider._refresh_weights()
    assert provider.weights[0] > provider.weights[1]


def test_never_visited_tracks_keep_a_nonzero_sampling_chance():
    pool = _pool(n=3)
    provider = RegretCurriculumTrackProvider(pool, refresh_every=1, seed=0)
    provider()
    provider.record_episode(10.0)
    provider._refresh_weights()
    assert all(w > 0 for w in provider.weights)
