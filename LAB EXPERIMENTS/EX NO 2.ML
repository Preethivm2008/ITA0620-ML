import csv

def candidate_elimination(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))

    attributes = data[0][:-1]
    examples = data[1:]

    S = ['0'] * len(attributes)
    G = [['?' for _ in attributes]]

    for example in examples:
        x = example[:-1]
        target = example[-1].strip().lower()

        if target == 'yes':
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(
                g[i] == '?' or g[i] == x[i]
                for i in range(len(x))
            )]

            # Generalize S
            for i in range(len(x)):
                if S[i] == '0':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'

        else:
            # Specialize G for a negative example
            new_G = []

            for g in G:
                if all(g[i] == '?' or g[i] == x[i]
                       for i in range(len(x))):
                    for i in range(len(x)):
                        if S[i] != '?' and S[i] != '0' \
                                and S[i] != x[i]:
                            h = g.copy()
                            h[i] = S[i]
                            new_G.append(h)
                else:
                    new_G.append(g)

            G = new_G

        # Remove duplicate hypotheses
        unique_G = []
        for g in G:
            if g not in unique_G:
                unique_G.append(g)
        G = unique_G

        print("\nTraining Example:", example)
        print("Specific Boundary S:", S)
        print("General Boundary G:", G)

    print("\nFinal Specific Boundary S:", S)
    print("Final General Boundary G:", G)


candidate_elimination("training_data.csv")