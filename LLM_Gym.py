import torch
from transformers import pipeline
from os import environ
from tqdm import trange
import gymnasium as gym
from dotenv import load_dotenv
from gymnasium.vector import SyncVectorEnv

load_dotenv()

def format_observation(d) -> str:
    # # i.e. Sum is 14, dealer is 6, no ace.
    return f"Sum is {d[0]}, dealer is {d[1]}, {'have' if bool(d[2]) else 'no'} ace"


def extract_action(response) -> int:
    digits = [char for char in response if char.isdigit()]
    if len(digits) == 0:
        raise ValueError("No action chosen")
    else:
        return int(digits[-1])

def llama(messages) -> str:
    prompt = llama_pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = llama_pipe(
        [prompt],
        batch_size=1,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    print(outputs[0][0]["generated_text"])
    response = outputs[0][0]["generated_text"].split("<|assistant|>\n")[-1]
    return response

llama_pipe = pipeline(
    "text-generation",
    token=environ["HF_TOKEN"],
    model="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
#     model_kwargs={"load_in_8bit": True},
)

env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode='human')

observation, info = env.reset(seed=42)
trajectories = []

for episode in trange(1):
    episode_messages = [
        {
            "role": "system",
            "content": """You are an expert single-player blackjack player. Every turn, you'll see your current sum, the dealer's showing card value (1 of 2), and whether you have a usable ace. You win by (1) not exceeding 21, and (2) exceeding the dealer's hand -- hidden+showing card values.

Decide whether to stick or hit by writing "Action: 0" or "Action: 1" respectively. You MUST decide your action and remember to write "Action:".""",
        }
    ]
    while True:
        # print(f"Observation: {observation}")

        message = format_observation(observation)
        episode_messages += [{"role": "user", "content": message}]

        response =  env.action_space.sample() # llama(episode_messages)
        action = extract_action(response)

        episode_messages += [{"role": "assistant", "content": str(action)}]
        # print(f"Action: {action}")

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"RESET {observation}, {reward}")
            observation, info = env.reset()
            trajectories.append((episode_messages, observation, reward))
            break

env.close()

avg_reward = sum([reward for _, _, reward in trajectories]) / len(trajectories)
print(avg_reward)

torch.cuda.empty_cache()

bs = 8
steps = 1000
seeds = [42 * n for n in range(bs)]
system_message = """You are an expert single-player blackjack player. Every turn, you'll see your current sum, the dealer's showing card value (1 of 2), and whether you have a usable ace. You win by (1) not exceeding 21, and (2) exceeding the dealer's hand -- hidden+showing card values.

Decide to stick or hit by answering "Action: 0" or "Action: 1" respectively. You MUST choose one of 2 options; none else."""

# Language Model Pipeline
# llama_pipe = pipeline("text-generation", token=userdata.get('HF_TOKEN'), model="meta-llama/Llama-2-7b-chat-hf", device_map="auto", model_kwargs={"load_in_8bit": True})
llama_pipe.tokenizer.pad_token_id = llama_pipe.model.config.eos_token_id


# Function to generate responses for a batch of messages
def llama_batch(messages_batch):
    prompts = [
        llama_pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    outputs = llama_pipe(
        prompts,
        batch_size=bs,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    responses = [output[0]["generated_text"].split(
        "\n")[-1] for output in outputs]
    return responses


# Function to format observations for all environments
def format_observations(observations):
    formatted_messages = []
    player_hand, dealer_showing, usable_ace = observations
    for A, B, C in zip(player_hand, dealer_showing, usable_ace):
        message = f"Your hand: {A}, Dealer's showing: {B}, Usable ace: {'yes' if C else 'no'}."
        formatted_messages.append(message)
    return formatted_messages


# Function to extract actions from responses
def extract_actions(responses):
    actions = []
    for response in responses:
        if "Action: 1" in response:
            actions.append(1)  # Hit
        elif "Action: 0" in response:
            actions.append(0)  # Stand or default action
        else:
            raise NotImplementedError(f'Generated invalid action "{response}"')
    return actions


# Create multiple environments
def make_env():
    return gym.make("Blackjack-v1", natural=False, sab=False)


envs = SyncVectorEnv([make_env for _ in range(bs)])
trajectories = []

# Run episodes
try:
    for episode in (pbar := trange(steps)):
        observations, infos = envs.reset()
        episode_messages_batch = [
            [{"role": "system", "content": system_message}] for _ in range(16)
        ]

        while True:
            messages_batch = format_observations(observations)
            episode_messages_batch = [
                episode_messages + [{"role": "user", "content": message}]
                for episode_messages, message in zip(
                    episode_messages_batch, messages_batch
                )
            ]

            responses = llama_batch(episode_messages_batch)
            actions = extract_actions(responses)

            episode_messages_batch = [
                episode_messages
                + [{"role": "assistant", "content": f"Action: {action}"}]
                for episode_messages, action in zip(episode_messages_batch, responses)
            ]

            observations, rewards, terminated, truncated, infos = envs.step(
                actions)

            # print(observations, '\n\n', rewards, '\n\n',  terminated, '\n\n', truncated, '\n\n', infos)

            if all(terminated) or all(truncated):
                # Access final observation and info if needed
                # final_observation = info.get('final_observation')
                # final_info = info.get('final_info')

                batch = [(x, y)
                         for x, y in zip(episode_messages_batch, rewards)]
                trajectories += batch
                avg_reward = sum([y for (_, y) in trajectories]
                                 ) / len(trajectories)
                pbar.set_postfix(win=avg_reward)

            # Check if all environments have terminated
            if all(terminated):
                break
finally:
    envs.close()

from datasets import Dataset

dataset = {
    "prompt": [x for (x, _) in trajectories],
    "label": [y for (_, y) in trajectories],
}

dataset = Dataset.from_dict(dataset).train_test_split(test_size=0.1)
print(type(dataset))

dataset.push_to_hub("photonmz/blackjack-gpt", token="YOUR_API_KEY")
