import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    # TODO: Compute the forward messages
    # Alpha(0)
    forward_messages[0] = rover.Distribution()
    for z0 in all_possible_hidden_states:
        forward_messages[0][z0] = observation_model(z0)[observations[0]] * prior_distribution[z0]
    forward_messages[0].renormalize()

    # Alpha_k(z_k) for k = 1 to n
    for n in range(1, num_time_steps):
        forward_messages[n] = rover.Distribution()
        for zk in all_possible_hidden_states:
            Pxkzk = observation_model(zk)[observations[n]] if (observations[n] != None) else 1

            sum = 0
            for zkm1 in forward_messages[n - 1]:
                alpha = forward_messages[n - 1][zkm1]
                hidden = transition_model(zkm1)[zk]
                sum += hidden * alpha

            forward_messages[n][zk] = Pxkzk * sum
        forward_messages[n].renormalize()  # normalize forward messages

    # TODO: Compute the backward messages

    # Beta(N-1) = 1
    backward_messages[num_time_steps - 1] = rover.Distribution()
    for zNm1 in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][zNm1] = 1
    # Beta_k(z_k) for k = N-2 to n
    for n in range(1, num_time_steps):
        k = num_time_steps - 1 - n
        backward_messages[k] = rover.Distribution()
        for zk in all_possible_hidden_states:
            sum = 0
            for zp1 in backward_messages[k+1]:
                Pxkp1zkp1 = observation_model(zp1)[observations[k+1]] if (observations[k+1] != None) else 1
                hidden = transition_model(zk)[zp1]
                beta = backward_messages[k+1][zp1]
                sum += beta * Pxkp1zkp1 * hidden
            if sum != 0:
                backward_messages[k][zk] = sum
        backward_messages[k].renormalize()

    # TODO: Compute the marginals

    for n in range(0, num_time_steps):
        marginals[n] = rover.Distribution()
        sum = 0
        for zk in all_possible_hidden_states:
            if forward_messages[n][zk] == 0 or backward_messages[n][zk] == 0:
                continue
            marginals[n][zk] = forward_messages[n][zk] * backward_messages[n][zk]
            sum += forward_messages[n][zk] * backward_messages[n][zk]

        for zk in marginals[n].keys():
            marginals[n][zk] = marginals[n][zk] / sum

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    z_prev = [None] * num_time_steps

    # w0
    w[0] = rover.Distribution()
    for z0 in all_possible_hidden_states:
        P_x0z0 = observation_model(z0)[observations[0]] if (observations[0] != None) else 1
        P_z0 = prior_distribution[z0]
        if P_z0 == 0 or P_x0z0 == 0:
            continue
        w[0][z0] = np.log(P_x0z0) + np.log(P_z0)

    # wk
    for k in range(1, num_time_steps):
        w[k] = rover.Distribution()
        z_prev[k] = {}
        for z_k in all_possible_hidden_states:
            P_xkzk = observation_model(z_k)[observations[k]] if (observations[k] != None) else 1
            max = -np.inf
            for z_km1 in w[k - 1]:
                P_zkzkm1 = transition_model(z_km1)[z_k]
                w_km1zm1 = w[k - 1][z_km1]
                if P_zkzkm1 == 0:
                    continue
                curr = np.log(P_zkzkm1) + w_km1zm1
                if curr > max:
                    max = curr
                    z_prev[k][z_k] = z_km1
            if P_xkzk != 0:
                w[k][z_k] = np.log(P_xkzk) + max

    max_w =  0 - np.inf
    for z_k in w[num_time_steps - 1]:
        curr = w[num_time_steps - 1][z_k]
        if curr > max_w:
            max_w = curr
            estimated_hidden_states[num_time_steps - 1] = z_k

    for k in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps - 1 - k] = z_prev[num_time_steps - k][estimated_hidden_states[num_time_steps - k]]

    return estimated_hidden_states

def compute_error(num_time_steps, hidden_states, estimated_states, marginals):
    # compute error
    error = 0
    for n in range(0, num_time_steps):
        if hidden_states[n] != estimated_states[n]:
            error += 1
    print("viterbi error:", error / 100)

    error = 0
    for n in range(0, num_time_steps):
        z = None
        prob = 0
        for zi in marginals[n]:
            if marginals[n][zi] > prob:
                z = zi
                prob = marginals[n][zi]
        if hidden_states[n] != z:
            error = error + 1
    print("fb error:", error / 100)

if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')



    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    compute_error(num_time_steps, hidden_states, estimated_states, marginals)
    for time_step in range(num_time_steps-1):
        state1 = estimated_states[time_step]
        state2 = estimated_states[time_step + 1]
        if state1[2] == state2[2] and state1[2] == 'stay' and (state1[0] != state2[0] or state1[1] != state2[1]):
            print("theres a goddamn problem!", state1, state2, time_step)
    print(estimated_states[64], estimated_states[65])
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

