import numpy as np

# import matplotlib.pyplot as plt

###############################################################################
###############################################################################
#                                 SWITCHER SIMULATIONS
###############################################################################
###############################################################################


def simul_hois(
    amplitude=1,
    target=False,
    target_frites=False,
    n_trials=1000,
    n_nodes=12,
    n_times=None,
    data_type="array",
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """
    Simulates High Order Interactions (HOIs) with or without behavioral
     information, depending on the 'target' parameter. The simulation
     can be conducted for a specified number of trials and nodes,
     with an optional time component.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        target (bool): Indicates whether to include behavioral
        information (True) or not (False). Default is False.
        target_frites (bool): Indicates whether to include
        behavioral information (True) or not (False). In such
        a way that is compatible witht the computation of
        behavioural variables as done in the FRITES toolbox
        Default is False.
        n_trials (int): Number of trials to simulate.
        Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        n_times (int or None): Number of time points in the
        simulated data. Default is None.
        If None, static HOIs are generated.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None or list): Time bump parameter. List
        of instant of time in which to have a bump.
        If None the bumps are equidistant, distributed
        in the time interval.
        time_length_bump (None or list): Time length bump
        parameter. A list containing in order the length of the
        time bumps. If None one is computed in such a way to have
        None overlapping bumps.
        triplet_character (list): List of triplet of desired HOIs.
        If None, half of the triplet is syn and
        the other half are red.
        triplet_character_with_beh (list): List of triplet
        characteristics with behavioral information.
        As triplet_character, but in relationship whit
        the target variables.

    Returns:
        numpy.ndarray or list: Simulated data. If n_times is provided,
        a list containing the simulated data,
        simulated data without behavioral information over time, and
        simulated data with behavioral
        information over time. Otherwise, a numpy array representing
        the simulated data.
        The shape of the generated data is either n_trails, n_nodes,
        n_times or n_trials, n_nodes

    """

    if n_times is None:
        if target:
            data_simulated = sim_hois_beh(
                amplitude=amplitude,
                n_trials=n_trials,
                n_nodes=n_nodes,
                data_type=data_type,
                triplet_character=triplet_character,
                triplet_character_with_beh=triplet_character_with_beh,
            )
        else:
            data_simulated = sim_hois(
                amplitude=amplitude,
                n_trials=n_trials,
                n_nodes=n_nodes,
                data_type=data_type,
                triplet_character=triplet_character,
            )

    else:
        if target:
            data_simulated = simulation_hois_beh(
                amplitude=amplitude,
                n_trials=n_trials,
                n_nodes=n_nodes,
                n_times=n_times,
                data_type=data_type,
                time_bump=time_bump,
                time_length_bump=time_length_bump,
                triplet_character=triplet_character,
                triplet_character_with_beh=triplet_character_with_beh,
            )

        elif target_frites:
            data_simulated = simulation_hois_beh_frites(
                amplitude=amplitude,
                n_trials=n_trials,
                n_nodes=n_nodes,
                n_times=n_times,
                data_type=data_type,
                time_bump=time_bump,
                time_length_bump=time_length_bump,
                triplet_character=triplet_character,
                triplet_character_with_beh=triplet_character_with_beh,
            )

        else:
            data_simulated = simulation_hois(
                amplitude=amplitude,
                n_trials=n_trials,
                n_nodes=n_nodes,
                n_times=n_times,
                data_type=data_type,
                time_bump=time_bump,
                time_length_bump=time_length_bump,
                triplet_character=triplet_character,
            )

    return data_simulated


###############################################################################
###############################################################################
#                                 TEMPORAL HOIs
###############################################################################
###############################################################################


def simulation_hois_beh(
    amplitude=1,
    n_trials=1000,
    n_nodes=12,
    n_times=100,
    data_type="array",
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """
    Simulates High Order Interactions (HOIs) with behavioral
    information over time.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        n_trials (int): Number of trials to simulate. Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        n_times (int): Number of time points in the simulated data.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None): Time bump parameter. Default is None.
        time_length_bump (None): Time length bump parameter.
        Default is None.
        triplet_character (list): List of triplet characteristics.
        Default is None.
        triplet_character_with_beh (list): List of triplet characteristics
        with behavioral information. Default is None.

    Returns:
        list: A list containing the simulated data, simulated data without
        behavioral information over time,
        and simulated data with behavioral information over time.
    """

    n_triplets = int(n_nodes / 3)

    if time_length_bump is None:
        time_length_bump = [
            int(n_times / n_triplets) for i in range(n_triplets)
        ]

    if time_bump is None:
        time_bump = []
        cc = int(time_length_bump[0] / 2)
        time_bump.append(cc)
        for tt in time_length_bump:
            cc = cc + tt
            time_bump.append(cc)

    mean_mvgauss, cov = cov_all_beh(
        n_nodes=n_nodes,
        triplet_character=triplet_character,
        triplet_character_with_beh=triplet_character_with_beh,
    )

    han_list = []
    for i, t_length in enumerate(time_length_bump):
        aa = np.zeros(n_times)
        aa[
            time_bump[i]
            - int(t_length / 2) : time_bump[i]
            - int(t_length / 2)
            + t_length
        ] = np.hanning(t_length)
        han_list.append(aa)

    simulated_data = np.zeros((n_trials, n_nodes + 1, n_times))

    cov_var = np.identity(n_nodes + 1)

    for t in range(n_times):
        for i, han in enumerate(han_list):
            cov_var[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = (
                np.ones((3, 3)) - np.identity(3)
            ) * han[t] + np.identity(3)
            cov_var[3 * i : 3 * (i + 1), 12] = (np.ones(3)) * han[t]
            cov_var[12, 3 * i : 3 * (i + 1)] = (np.ones(3)) * han[t]

        simulated_data[:, :, t] = np.random.multivariate_normal(
            mean_mvgauss,
            cov * cov_var,
            size=n_trials,
            check_valid="warn",
            tol=1e-8,
        )

    return [
        simulated_data,
        simulated_data[:, :12, :],
        simulated_data[:, 12, :],
    ]


def simulation_hois_beh_frites(
    amplitude=1,
    n_trials=1000,
    n_nodes=12,
    n_times=100,
    data_type="array",
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """
    Simulates High Order Interactions (HOIs) with behavioral
    information over time.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        n_trials (int): Number of trials to simulate. Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        n_times (int): Number of time points in the simulated data.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None): Time bump parameter. Default is None.
        time_length_bump (None): Time length bump parameter. Default is None.
        triplet_character (list): List of triplet characteristics.
        Default is None.
        triplet_character_with_beh (list): List of triplet characteristics with
        behavioral information. Default is None.

    Returns:
        list: A list containing the simulated data, simulated data
        without behavioral information over time, and simulated
        data with behavioral information over time.

    """

    n_triplets = int(n_nodes / 3)

    if time_length_bump is None:
        time_length_bump = [
            int(n_times / n_triplets) for i in range(n_triplets)
        ]

    if time_bump is None:
        time_bump = []
        cc = int(time_length_bump[0] / 2)
        time_bump.append(cc)
        for tt in time_length_bump:
            cc = cc + tt
            time_bump.append(cc)

    mean_mvgauss, cov_al = cov_all_beh(
        n_nodes=n_nodes,
        triplet_character=triplet_character,
        triplet_character_with_beh=triplet_character_with_beh,
    )

    han_list = []
    for i, t_length in enumerate(time_length_bump):
        aa = np.zeros(n_times)
        aa[
            time_bump[i]
            - int(t_length / 2) : time_bump[i]
            - int(t_length / 2)
            + t_length
        ] = np.hanning(t_length)
        aa_temp = np.tile(
            aa[np.newaxis, np.newaxis, :], (n_trials, n_nodes, 1)
        )
        han_list.append(aa_temp)

    han_inverse_list = []
    for i, t_length in enumerate(time_length_bump):
        aa = np.ones(n_times)
        aa[
            time_bump[i]
            - int(t_length / 2) : time_bump[i]
            - int(t_length / 2)
            + t_length
        ] = 1 - np.hanning(t_length)
        aa_temp = np.tile(
            aa[np.newaxis, np.newaxis, :], (n_trials, n_nodes, 1)
        )
        han_inverse_list.append(aa_temp)

    sim_static_rand = np.random.multivariate_normal(
        mean_mvgauss,
        cov=np.identity(n_nodes + 1),
        size=n_trials,
        check_valid="warn",
        tol=1e-8,
    )
    sim_rand = np.tile(sim_static_rand[:, :, np.newaxis], (1, 1, n_times))

    sim_static_hois = np.random.multivariate_normal(
        mean_mvgauss, cov=cov_al, size=n_trials, check_valid="warn", tol=1e-8
    )
    sim_hois = np.tile(sim_static_hois[:, :, np.newaxis], (1, 1, n_times))

    for i, aaa in enumerate(zip(han_list, han_inverse_list)):
        han, han_inv = aaa
        sim_rand[:, i * 3 : (i + 1) * 3, :] = (
            sim_hois[:, i * 3 : (i + 1) * 3, :]
            * han[:, i * 3 : (i + 1) * 3, :]
            + sim_rand[:, i * 3 : (i + 1) * 3, :]
            * han_inv[:, i * 3 : (i + 1) * 3, :]
        )

    simulated_data = sim_rand

    return [simulated_data[:, :12, :], sim_hois[:, 12, 0]]


def simulation_hois(
    amplitude=1,
    n_trials=1000,
    n_nodes=12,
    n_times=100,
    data_type="array",
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
):
    """
    Simulates High Order Interactions (HOIs) without behavioral
    information over time.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        n_trials (int): Number of trials to simulate. Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        n_times (int): Number of time points in the simulated data.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None): Time bump parameter. Default is None.
        time_length_bump (None): Time length bump parameter. Default is None.
        triplet_character (list): List of triplet characteristics.
        Default is None.

    Returns:
        numpy.ndarray: Simulated data.

    """

    n_triplets = int(n_nodes / 3)

    if time_length_bump is None:
        time_length_bump = [
            int(n_times / n_triplets) for i in range(n_triplets)
        ]

    if time_bump is None:
        time_bump = []
        cc = int(time_length_bump[0] / 2)
        time_bump.append(cc)
        for tt in time_length_bump:
            cc = cc + tt
            time_bump.append(cc)

    mean_mvgauss, cov = cov_all(
        n_nodes=n_nodes,
        triplet_character=triplet_character,
    )

    han_list = []
    for i, t_length in enumerate(time_length_bump):
        aa = np.zeros(n_times)
        aa[
            time_bump[i]
            - int(t_length / 2) : time_bump[i]
            - int(t_length / 2)
            + t_length
        ] = np.hanning(t_length)
        han_list.append(aa)

    simulated_data = np.zeros((n_trials, n_nodes, n_times))

    cov_var = np.identity(n_nodes)

    for t in range(n_times):
        for i, han in enumerate(han_list):
            cov_var[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = (
                np.ones((3, 3)) - np.identity(3)
            ) * han[t] + np.identity(3)

        simulated_data[:, :, t] = np.random.multivariate_normal(
            mean_mvgauss,
            cov * cov_var,
            size=n_trials,
            check_valid="warn",
            tol=1e-8,
        )

    return simulated_data


###############################################################################
###############################################################################
#                                   STATIC HOIs
###############################################################################
###############################################################################


def sim_hois_beh(
    amplitude=1,
    n_trials=1000,
    n_nodes=12,
    data_type="array",
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """
    Simulates High Order Interactions (HOIs) with behavioral information.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        n_trials (int): Number of trials to simulate. Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        data_type (str): Type of data. Default is 'array'.
        triplet_character (list): List of triplet characteristics.
        Default is None.
        triplet_character_with_beh (list): List of triplet characteristics with
        behavioral information. Default is None.

    Returns:
        list: A list containing the simulated data, simulated data without
        behavioral information, and simulated data with behavioral information.

    """

    # n_triplets = int(n_nodes/3)

    mean_mvgauss, cov = cov_all_beh(
        n_nodes=n_nodes,
        triplet_character=triplet_character,
        triplet_character_with_beh=triplet_character_with_beh,
    )

    simulated_data = np.zeros((n_trials, n_nodes + 1))

    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_trials, check_valid="warn", tol=1e-8
    )

    return [simulated_data, simulated_data[:, :12], simulated_data[:, 12]]


def sim_hois(
    amplitude=1,
    n_trials=1000,
    n_nodes=12,
    data_type="array",
    triplet_character=None,
):
    """
    Simulates High Order Interactions (HOIs) without behavioral information.

    Parameters:
        amplitude (int): Amplitude parameter. Default is 1.
        n_trials (int): Number of trials to simulate. Default is 1000.
        n_nodes (int): Number of nodes in the simulated data.
        data_type (str): Type of data. Default is 'array'.
        triplet_character (list): List of triplet characteristics.
        Default is None.

    Returns:
        numpy.ndarray: Simulated data.

    """

    # n_triplets = int(n_nodes/3)

    mean_mvgauss, cov = cov_all(
        n_nodes=n_nodes,
        triplet_character=triplet_character,
    )

    simulated_data = np.zeros((n_trials, n_nodes))

    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_trials, check_valid="warn", tol=1e-8
    )

    return simulated_data


###############################################################################
###############################################################################
#                        COVARIANCE OF n VARIABLES WITH HOIs
###############################################################################
###############################################################################


def cov_all_beh(
    n_nodes=12,
    data_type="array",
    time_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """
    Computes the mean and covariance matrix for a multivariate Gaussian
    distribution with behavioral information.

    Parameters:
        n_nodes (int): Number of nodes in the multivariate Gaussian
        distribution.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None): Time bump parameter. Default is None.
        triplet_character (list): List of triplet characteristics.
        Default is None.
        triplet_character_with_beh (list): List of triplet
        characteristics with behavioral information. Default is None.

    Returns:
        tuple: A tuple containing the mean and covariance matrix
        of the multivariate Gaussian distribution.

    """

    n_triplets = int(n_nodes / 3)

    if triplet_character is None:
        triplet_character = []
        for i in range(n_triplets):
            if (i % 2) == 0:
                triplet_character.append("synergy")
            else:
                triplet_character.append("redundancy")

    if triplet_character_with_beh is None:
        triplet_character_with_beh = []
        for i in range(int(n_triplets / 2)):
            if (i % 2) == 0:
                triplet_character_with_beh.append("synergy")
            else:
                triplet_character_with_beh.append("redundancy")

    mean_mvgauss = np.zeros(n_nodes + 1)

    # HOI params
    cov = np.zeros((n_nodes + 1, n_nodes + 1))

    for i, char in enumerate(triplet_character):
        cov[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = cov_order_3(char)

    for i, char in enumerate(triplet_character_with_beh):
        cov[3 * i : 3 * (i + 1), n_nodes] = cov_order_4(char)[3, :3]
        cov[n_nodes, 3 * i : 3 * (i + 1)] = cov_order_4(char)[3, :3]

    return mean_mvgauss, cov


def cov_all(
    n_nodes=12,
    data_type="array",
    time_bump=None,
    triplet_character=None,
):
    """
    Computes the mean and covariance matrix for a multivariate
    Gaussian distribution without behavioral information.

    Parameters:
        n_nodes (int): Number of nodes in the multivariate
        Gaussian distribution.
        data_type (str): Type of data. Default is 'array'.
        time_bump (None): Time bump parameter. Default is None.
        triplet_character (list): List of triplet characteristics.
        Default is None.

    Returns:
        tuple: A tuple containing the mean and covariance matrix of
        the multivariate Gaussian distribution.

    """

    n_triplets = int(n_nodes / 3)

    if triplet_character is None:
        triplet_character = []
        for i in range(n_triplets):
            if (i % 2) == 0:
                triplet_character.append("synergy")
            else:
                triplet_character.append("redundancy")

    mean_mvgauss = np.zeros(n_nodes)

    # HOI params
    cov = np.zeros((n_nodes, n_nodes))

    for i, char in enumerate(triplet_character):
        cov[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = cov_order_3(char)

    return mean_mvgauss, cov


###############################################################################
###############################################################################
#                        COVARIANCE WITH HOI, ORDER 3 & 4
###############################################################################
###############################################################################


def cov_order_3(character):
    """
    Compute the covariance matrix for three brain regions based
    on the given character.

    INPUTS:
    - character: 'null', 'redundancy', or 'synergy' indicating
    the relationship between brain regions

    OUTPUTS:
    - cov: covariance matrix for the three brain regions
    """

    lambx = np.sqrt(0.99)
    lamby = np.sqrt(0.7)
    lambz = np.sqrt(0.3)

    # Full factor matrix m
    m = np.array([lambx, lamby, lambz])[np.newaxis]

    if character == "null":
        # We fix theta_yz in such a way that O(R1,R2,R3)=0
        theta_yz = -0.148

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three brain regions
        cov = m * m.T + theta

    elif character == "redundancy":
        # We fix theta_yz in such a way that O(R1,R2,R3)>0
        theta_yz = -0.39

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three brain regions
        cov = m * m.T + theta

    elif character == "synergy":
        # We fix theta_yz in such a way that O(R1,R2,R3)<0
        theta_yz = 0.22

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three brain regions
        cov = m * m.T + theta

    return cov


def cov_order_4(character):
    """
    Calculate the covariance matrix for a given character.

    Parameters:
        character (str): The character specifying the type of covariance
        matrix. It can be either 'redundancy' or 'synergy'.

    Returns:
        cov_ (ndarray): The covariance matrix for the specified character.
    """
    lambx = np.sqrt(0.99)
    lamby = np.sqrt(0.7)
    lambz = np.sqrt(0.3)
    lambs = np.sqrt(0.2)

    # Imposing the relationships with the behavior
    m = np.array([lambx, lamby, lambz, lambs])[np.newaxis]

    theta = np.diagflat(1 - m**2)

    if character == "redundancy":
        theta_zs = 0.25
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )
        cov_ = m * m.T + theta

    if character == "synergy":
        theta_zs = -0.52
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )
        cov_ = m * m.T + theta

    return cov_


def simulate_hois_gauss(
    n_samples=200,
    n_features=4,
    n_variables=1,
    triplets=[(0, 1, 2), (1, 2, 3)],
    tiplet_types=["redundancy", "synergy"],
):
    """Simulate higher-order interactions.

    n_samples: number of samples
    n_features: number of nodes
    n_variables: number of repetitions. This parameter can be used
    to simulate dynamic hoi
    triplets: list of triplets of nodes linked by hoi.
    tiplet_types: specify whether each triplet interaction type
    should be redundant or synergistic. By default
    the function generates redundant interactions between
    nodes (0, 1, 2) and synergistic interactions between
    nodes (1, 2, 3).
    """
    pass


if __name__ == "__main__":
    from hoi.metrics import Oinfo
    from hoi.utils import get_nbest_mult

    # simulate hois
    x = simulate_hois_gauss()

    # compute hois
    oi = np.zeros((3, 4))  # x
    model = Oinfo(x=oi)
    hoi = model.fit(x)

    # print the results and check that it correspond to the ground truth
    df = get_nbest_mult(hoi, model=model)
