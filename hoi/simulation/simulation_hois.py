import numpy as np

# import matplotlib.pyplot as plt

###############################################################################
###############################################################################
#                                 SWITCHER SIMULATIONS
###############################################################################
###############################################################################


def simulate_hois_gauss(
    target=False,
    n_trials=200,
    n_nodes=12,
    n_times=None,
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """Simulates High Order Interactions (HOIs) with or without behavioral
    information, depending on the 'target' parameter.

    The simulation can be conducted for a specified number of trials and nodes,
    with an optional time component.

    Parameters
    ----------
    target : bool | False
        Indicates whether to include behavioral
        information (True) or not (False).
    n_trials : int | 1000
        Number of trials to simulate.
    n_nodes : int | 12
        Number of nodes in the simulated data.
    n_times : int | None
        Number of time points in the
        simulated data. If None, static HOIs are generated.
    time_bump : list | None
        Time bump parameter. List of instant of time in
        which to have a bump. If None the bumps are equidistant,
        distributed in the time interval.
    time_length_bump : list | None
        Time length bump parameter. A list containing in order the
        length of the time bumps. If None one is computed in such
        a way to have None overlapping bumps.
    triplet_character : list | None
        List of triplet of desired HOIs.
        If None, half of the triplet is syn and
        the other half are red.
    triplet_character_with_beh : list | None
        List of triplet characteristics with behavioral information.
        As triplet_character, but in relationship whit the target
        variables.

    Returns
    -------
    Simulated data : numpy.ndarray
        If n_times is provided, a list containing the simulated data,
        simulated data without behavioral information over time, and
        simulated data with behavioral
        information over time. Otherwise, a numpy array representing
        the simulated data.
        The shape of the generated data is either n_trails, n_nodes,
        n_times or n_trials, n_nodes

    """

    if not isinstance(n_times, int) and not target:
        # static without target
        return sim_hoi_static(
            n_trials=n_trials,
            n_nodes=n_nodes,
            triplet_character=triplet_character,
        )

    elif not isinstance(n_times, int) and target:
        # static with target
        return sim_hoi_static_target(
            n_trials=n_trials,
            n_nodes=n_nodes,
            triplet_character=triplet_character,
            triplet_character_with_beh=triplet_character_with_beh,
        )

    elif isinstance(n_times, int) and not target:
        # dynamic without target
        return sim_hoi_dyn(
            n_trials=n_trials,
            n_nodes=n_nodes,
            n_times=n_times,
            time_bump=time_bump,
            time_length_bump=time_length_bump,
            triplet_character=triplet_character,
        )

    elif isinstance(n_times, int) and target:
        # dynamic with target
        return sim_hoi_dyn_target(
            n_trials=n_trials,
            n_nodes=n_nodes,
            n_times=n_times,
            time_bump=time_bump,
            time_length_bump=time_length_bump,
            triplet_character=triplet_character,
            triplet_character_with_beh=triplet_character_with_beh,
        )


###############################################################################
###############################################################################
#                                 TEMPORAL HOIs
###############################################################################
###############################################################################


def sim_hoi_dyn_target(
    n_trials=1000,
    n_nodes=12,
    n_times=100,
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """Simulates High Order Interactions (HOIs) with behavioral
    information over time.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    n_nodes :int | 12
        Number of nodes in the simulated data.
    n_times : int | 100
        Number of time points in the simulated data.
    time_bump : None
        Time bump parameter.
    time_length_bump : None
        Time length bump parameter.
    triplet_character : list | None
        List of triplet characteristics.
    triplet_character_with_beh : list | None
        List of triplet characteristics with
        behavioral information.

    Returns
    -------
        Simulated data : numpy.ndarray
        simulated data without behavioral information over time,
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

    return simulated_data[:, :12, :], sim_hois[:, 12, 0]


def sim_hoi_dyn(
    n_trials=1000,
    n_nodes=12,
    n_times=100,
    time_bump=None,
    time_length_bump=None,
    triplet_character=None,
):
    """Simulates High Order Interactions (HOIs) without behavioral
    information over time.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    n_nodes : int | 12
        Number of nodes in the simulated data.
    n_times : int | 100
        Number of time points in the simulated data.
    time_bump : None
        Time bump parameter.
    time_length_bump : None
        Time length bump parameter.
    triplet_character : list | None
        List of triplet characteristics.

    Returns
    ------
        Simulated data : numpy.ndarray

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


def sim_hoi_static_target(
    n_trials=1000,
    n_nodes=12,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """Simulates High Order Interactions (HOIs) with behavioral information.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    n_nodes : int | 12
        Number of nodes in the simulated data.
    triplet_character : list | None
        List of triplet characteristics.
    triplet_character_with_beh : list | None
        List of triplet characteristics with
        behavioral information.

    Returns
    -------
        simulated data, simulated data without behavioral information,
        simulated data with behavioral information : numpy.ndarray

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

    return simulated_data, simulated_data[:, :12], simulated_data[:, 12]


def sim_hoi_static(
    n_trials=1000,
    n_nodes=12,
    triplet_character=None,
):
    """Simulates High Order Interactions (HOIs) without behavioral information.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    n_nodes : int | 12
        Number of nodes in the simulated data.
    triplet_character : list | None
        List of triplet characteristics.

    Returns
    -------
    Simulated data : numpy.ndarray

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
    time_bump=None,
    triplet_character=None,
    triplet_character_with_beh=None,
):
    """Computes the mean and covariance matrix for a multivariate Gaussian
    distribution with behavioral information.

    Parameters
    ----------
    n_nodes : int | 12
        Number of nodes in the multivariate Gaussian distribution.
    time_bump | None
        Time bump parameter.
    triplet_character : list | None
        List of triplet characteristics.
    triplet_character_with_beh : list | None
        List of triplet characteristics with behavioral information.

    Returns
    -------
    mean and covariance matrix of the multivariate Gaussian
    distribution : numpy.ndarray

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
    time_bump=None,
    triplet_character=None,
):
    """Computes the mean and covariance matrix for a multivariate
    Gaussian distribution without behavioral information.

    Parameters
    ----------
    n_nodes : int | 12
        Number of nodes in the multivariate
        Gaussian distribution.
    time_bump : None
        Time bump parameter.
    triplet_character : list | None
        List of triplet characteristics.

    Returns
    -------
    mean and covariance matrix of the multivariate Gaussian
    distribution : numpy.ndarray

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
    """Compute the covariance matrix for three brain regions based
    on the given character.

    Parameters
    ----------
    character : str
        'null', redundancy', or 'synergy' indicating the relationship
        between brain regions

    Returns
    -------
    cov : numpy.ndarray
        covariance matrix for the three brain regions
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
    """Calculate the covariance matrix for a given character.

    Parameters
    ----------
    character : str
        The character specifying the type of covariance
        matrix. It can be either 'redundancy' or 'synergy'.

    Returns
    -------
    cov_ : numpy.ndarray
    The covariance matrix for the specified character.
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
