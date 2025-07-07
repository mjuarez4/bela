import torch

stats = torch.load("dataset_stats_m.pt", map_location="cpu")
print(stats.keys())  # should be ['human', 'robot']
print(type(stats["human"]))  # e.g., DataStats
print(stats["human"].__dict__.keys())  # attributes inside DataStats

