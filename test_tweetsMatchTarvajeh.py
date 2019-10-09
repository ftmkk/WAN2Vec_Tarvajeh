from Utils.WAN import WAN

wan = WAN('inputs/allResponses3.csv')


with open('inputs/tweets_tokens.txt') as input_f:
    cntr = 0
    plus_cntr = 0
    while True:
        cntr += 1
        # if cntr % 10000 == 0:
        #     print(cntr)
        line = input_f.readline()
        if not line:
            break

        tokens = line.split(' ')
        tokens_in_wan = [token for token in tokens if token in wan.nodes]
        if len(tokens_in_wan) >= 7:
            plus_cntr += 1
    print(plus_cntr)
    print(cntr)
