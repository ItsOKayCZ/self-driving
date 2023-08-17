import math

class Scheduler:
	def __init__(self, num_epochs, starting_exploration, num_resets=5):
		self.num_epochs = num_epochs
		self.starting_exploration = starting_exploration
		self.num_resets = num_resets

		self.exploration = starting_exploration

		self.exploration_reduce = self.starting_exploration / math.floor(self.num_epochs / self.num_resets)

	def step(self):
		if self.exploration <= 0:
			self.exploration = self.starting_exploration
		else:
			self.exploration -= self.exploration_reduce
			self.exploration = max(0, self.exploration)
		
		return self.exploration