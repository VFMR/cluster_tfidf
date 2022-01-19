from .utils import cosine_sim_words

def check_pairs(lst, embeddings):
    for row in lst:
        for a in row:
            for b in row:
                if a!=b:
                    print(f'{a} & {b}: {cosine_sim_words(a, b, embeddings):0.2f}')



def analyze_clusters(clustering, filename):
    index2word = clustering.index2word
    index2cluster = clustering.index2cluster

    multi_clusters = clustering.get_multi_clusters()
    print(f'Final number of clusters: {len(multi_clusters)}')

    words = {y: [index2word[x] for x in [key for key, value in index2cluster.items() if value==y]] for y in multi_clusters}
    with open(filename, 'w') as f:
        for key, value in words.items():
            f.write(f'{key}: {value}\n')
    return words

