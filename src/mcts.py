import warnings
warnings.filterwarnings("ignore")

import numpy as np
from typing import List, Optional, Generator
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) structure.
    """

    def __init__(self, state: torch.Tensor, parent: Optional["Node"] = None) -> None:
        """
        Initialize a Node.

        :param state: The state or token IDs associated with this Node.
        :param parent: The parent Node of this Node; None if this Node is the root.
        """
        self.state: torch.Tensor = state
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
        self.visits: int = 0
        self.value: float = 0.0


class MCTS:
    """
    A Monte Carlo Tree Search (MCTS) utility class that leverages a pretrained model
    for text generation and exploration.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_depth: int = 10,
        num_simulations: int = 10,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the MCTS module.

        :param model: The pretrained model used for token generation.
        :param tokenizer: The tokenizer paired with the pretrained model.
        :param max_depth: Maximum depth for the search.
        :param num_simulations: Number of simulations to run for search.
        :param max_new_tokens: Maximum tokens to generate during simulation.
        :param temperature: Sampling temperature used during generation.
        :param stop_tokens: Optional list of tokens that will halt generation.
        :param dtype: Data type to use for model inputs.
        :param device: Device to use for model inputs.
        """
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_depth: int = max_depth
        self.num_simulations: int = num_simulations
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.stop_tokens: List[str] = stop_tokens if stop_tokens else []
        self.dtype: torch.dtype = dtype
        self.device: str = device

    def search(self, input_ids: torch.Tensor) -> Generator[torch.Tensor, None, torch.Tensor]:
        """
        Conduct the MCTS search and return the best child node.

        :param input_ids: The initial tokens/ids for the root node.
        :return: The best child node from the search process.
        """
        root = Node(input_ids)
        for _ in range(self.num_simulations):
            node = self.selection(root)
            node = self.expansion(node)
            reward = self.simulation(node)
            self.backpropagation(node, reward)

        best_child = self.get_best_child(root).state

        if len(best_child.shape) == 1:
            best_child = best_child.unsqueeze(0)

        return best_child
    
    def selection(self, node: Node) -> Node:
        """
        Traverse the tree to find a leaf Node by applying the UCB1 formula.

        :param node: The current node to start the selection process.
        :return: The selected leaf node.
        """
        if not node.children or node.visits == 0:
            return node
        best_child = None
        best_score = -np.inf
        for child in node.children:
            exploration_term = np.sqrt(2 * np.log(node.visits) / child.visits) if child.visits > 0 else 0.0
            score = (child.value / child.visits if child.visits > 0 else 0.0) + exploration_term
            if score > best_score:
                best_score = score
                best_child = child
        return self.selection(best_child)  # type: ignore

    def expansion(self, node: Node) -> Node:
        """
        Expand the node by creating child nodes based on model predictions.

        :param node: The node to expand.
        :return: The expanded node.
        """
        if node.children:
            return node

        if len(node.state.shape) == 1:
            node.state = node.state.unsqueeze(0)

        with torch.autocast(dtype=self.dtype, device_type=self.device):
            with torch.no_grad():
                outputs = self.model.generate(
                    node.state,
                    attention_mask=None,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.max_depth,
                    use_cache=True,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        for output in outputs:
            child = Node(output, parent=node)
            node.children.append(child)
        return node

    def simulation(self, node: Node) -> float:
        """
        Simulate the rollout from the current node using a forward pass of the model.
        Returns a reward value based on the model prediction or heuristic.

        :param node: The node to simulate from.
        :return: The reward value from the simulation.
        """
        if not node.children:
            return 0.0
        child = np.random.choice(node.children)
        reward = self.evaluate(child.state)
        return reward

    def backpropagation(self, node: Node, reward: float) -> None:
        """
        Backpropagate the simulation's reward value up the tree.

        :param node: The node to start backpropagation from.
        :param reward: The reward value to propagate.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def get_best_child(self, node: Node) -> Node:
        """
        Return the best child Node based on average value.

        :param node: The node to get the best child from.
        :return: The best child node.
        """
        best_child = None
        best_score = -np.inf
        for child in node.children:
            score = 0 if child.visits == 0 else child.value / child.visits
            if score > best_score:
                best_child = child
                best_score = score
        return best_child

    def evaluate(self, input_ids: torch.Tensor) -> float:
        """
        Evaluate the given input_ids and return a reward value.

        :param input_ids: The input token IDs to evaluate.
        :return: The reward value.
        """

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        return -loss.item()

    def has_stop_token(self, input_ids: torch.Tensor) -> bool:
        """
        Check if the input_ids contain any stop tokens.

        :param input_ids: The input token IDs to check.
        :return: True if any stop token is found, False otherwise.
        """
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_token in self.stop_tokens:
            if stop_token in decoded_text:
                return True
        return False
