from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TypedDict

from tlcs.agent import Agent
from tlcs.constants import TESTING_SETTINGS_FILE, TRAINING_SETTINGS_FILE
from tlcs.env import Environment, EnvStats
from tlcs.episode import Record, run_episode
from tlcs.logger import get_logger
from tlcs.memory import Memory, Sample
from tlcs.plots import save_data_and_plot
from tlcs.settings import load_testing_settings, load_training_settings

logger = get_logger(__name__)


class TrainingStats(TypedDict):
    """Aggregated statistics collected during training episodes."""

    sum_neg_reward: list[float]
    cumulative_wait: list[int]
    avg_queue_length: list[float]


class TestingStats(TypedDict):
    """Statistics collected during a testing episode."""

    reward: list[float]
    queue_length: list[int]


def add_experience_to_memory(memory: Memory, history: list[Record]) -> None:
    for i in range(len(history) - 1):
        sample = Sample(
            state=history[i].state,
            action=history[i].action,
            reward=history[i].reward,
            next_state=history[i + 1].state,
        )
        memory.add_sample(sample)


def update_training_stats(
    episode_history: list[Record],
    env_stats: list[EnvStats],
    max_steps: int,
    training_stats: TrainingStats,
) -> TrainingStats:
    sum_neg_reward = sum(record.reward for record in episode_history if record.reward < 0)
    training_stats["sum_neg_reward"].append(sum_neg_reward)

    sum_queue_length = sum(stats.queue_length for stats in env_stats)
    avg_queue_length = round(sum_queue_length / max_steps, 1)
    training_stats["avg_queue_length"].append(avg_queue_length)

    training_stats["cumulative_wait"].append(sum_queue_length)
    return training_stats


def training_session(settings_file: Path, out_path: Path) -> None:
    settings = load_training_settings(settings_file)

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)
    agent = Agent(settings=settings)

    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes

    training_stats: TrainingStats = {
        "sum_neg_reward": [],
        "cumulative_wait": [],
        "avg_queue_length": [],
    }

    # --- NEW: tracking best & worst ---
    best_reward = -float("inf")
    worst_reward = float("inf")
    save_dir = Path("Sumo_Trained_Simulations")
    save_dir.mkdir(exist_ok=True)
    route_file = Path("intersection/episode_routes.rou.xml")
    # ---------------------------------

    for episode in range(tot_episodes):
        logger.info(f"Episode {episode + 1} of {tot_episodes}")

        new_epsilon = round(1.0 - (episode / tot_episodes), 2)
        agent.set_epsilon(new_epsilon)

        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=episode)

        add_experience_to_memory(memory=memory, history=episode_history)

        for _ in range(settings.training_epochs):
            agent.replay(
                memory=memory,
                gamma=settings.gamma,
                batch_size=settings.batch_size,
            )

        training_stats = update_training_stats(
            episode_history=episode_history,
            env_stats=env_stats,
            max_steps=settings.max_steps,
            training_stats=training_stats,
        )

        last_neg_reward = training_stats["sum_neg_reward"][-1]
        last_cumulative_wait = training_stats["cumulative_wait"][-1]
        last_avg_queue_length = training_stats["avg_queue_length"][-1]

        logger.info(f"\tEpsilon: {agent.epsilon}")
        logger.info(f"\tReward: {last_neg_reward}")
        logger.info(f"\tCumulative wait: {last_cumulative_wait}")
        logger.info(f"\tAvg queue: {last_avg_queue_length}")

        # --- NEW: save best & worst ---
        if last_neg_reward > best_reward:
            best_reward = last_neg_reward
            copyfile(route_file, save_dir / "best_episode.rou.xml")
            logger.info("Saved BEST episode")

        if last_neg_reward < worst_reward:
            worst_reward = last_neg_reward
            copyfile(route_file, save_dir / "worst_episode.rou.xml")
            logger.info("Saved WORST episode")
        # --------------------------------

    out_path.mkdir(parents=True, exist_ok=True)
    agent.save_model(out_path)

    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Session info saved at: {out_path}")

    copyfile(src=settings_file, dst=out_path / TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["sum_neg_reward"],
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["cumulative_wait"],
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue_length"],
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
        out_folder=out_path,
    )


def testing_session(settings_file: Path, model_path: Path, test_name: str) -> None:
    settings = load_testing_settings(settings_file)

    test_path = model_path / test_name
    test_path.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        settings=load_training_settings(model_path / TRAINING_SETTINGS_FILE),
        epsilon=0,
        model_path=model_path,
    )

    env = Environment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        gui=settings.gui,
        sumocfg_file=settings.sumocfg_file,
    )

    episode_history, env_stats = run_episode(
        env=env,
        agent=agent,
        seed=settings.episode_seed,
    )

    testing_stats: TestingStats = {
        "reward": [record.reward for record in episode_history],
        "queue_length": [stats.queue_length for stats in env_stats],
    }

    copyfile(src=settings_file, dst=test_path / TESTING_SETTINGS_FILE)

    save_data_and_plot(
        data=testing_stats["reward"],
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        out_folder=test_path,
    )
    save_data_and_plot(
        data=testing_stats["queue_length"],
        filename="queue",
        xlabel="Step",
        ylabel="Queue length (vehicles)",
        out_folder=test_path,
    )

    logger.info(f"Testing results saved at: {test_path}")
