import torch
from torch_geometric.data import HeteroData
import numpy as np

def load_synthetic_twibot():
    """
    Creates a synthetic Heterogeneous PyG Graph structured identically to TwiBot-22.
    - Nodes: 'user', 'tweet', 'hashtag'
    - Edges: ('user', 'follows', 'user'), ('user', 'tweeted', 'tweet'), 
             ('tweet', 'retweeted', 'tweet'), ('tweet', 'contains', 'hashtag')
    """
    data = HeteroData()

    # Configuration for Mock sizes (scales well for visualization without crashing GUI)
    num_users = 200
    num_tweets = 500
    num_hashtags = 30

    # 1. Define Node Features
    # TwiBot users typically have textual, numerical, and categorical properties derived from BERT/ResNet.
    data['user'].x = torch.randn((num_users, 768)) # Simulated BERT embeddings
    data['user'].y = torch.randint(0, 2, (num_users,)) # Bot vs Human
    
    data['tweet'].x = torch.randn((num_tweets, 768)) 
    data['hashtag'].x = torch.randn((num_hashtags, 128)) 

    # 2. Define Explicit Heterogeneous Edges
    # Users follow Users (Homogeneous edge inside Heterogeneous graph)
    user_to_user_src = torch.randint(0, num_users, (600,))
    user_to_user_dst = torch.randint(0, num_users, (600,))
    data['user', 'follows', 'user'].edge_index = torch.stack([user_to_user_src, user_to_user_dst], dim=0)

    # Users post Tweets (Bipartite edge)
    user_to_tweet_src = torch.randint(0, num_users, (num_tweets,))
    # Every tweet has exactly one author, so dst is just arange
    user_to_tweet_dst = torch.arange(0, num_tweets)
    data['user', 'tweeted', 'tweet'].edge_index = torch.stack([user_to_tweet_src, user_to_tweet_dst], dim=0)

    # Tweets contain Hashtags (Tripartite mapping)
    tweet_to_hash_src = torch.randint(0, num_tweets, (800,))
    tweet_to_hash_dst = torch.randint(0, num_hashtags, (800,))
    data['tweet', 'contains', 'hashtag'].edge_index = torch.stack([tweet_to_hash_src, tweet_to_hash_dst], dim=0)

    # Metadata for stats
    info = {
        'dataset': 'TwiBot-22-Synthetic',
        'task': 'bot',
        'num_users': num_users,
        'num_tweets': num_tweets,
        'num_hashtags': num_hashtags,
        'human_count': (data['user'].y == 0).sum().item(),
        'bot_count': (data['user'].y == 1).sum().item()
    }
    
    return data, info

if __name__ == "__main__":
    data, info = load_synthetic_twibot()
    print("Heterogeneous TwiBot-22 Mock Generated:")
    print(data)
    print("Metadata:", info)
