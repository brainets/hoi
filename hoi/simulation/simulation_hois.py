import numpy as np

# import matplotlib.pyplot as plt

###############################################################################
###############################################################################
#                                 SWITCHER SIMULATIONS
###############################################################################
###############################################################################


def simulate_hois_gauss(
    target=False,
    n_trials=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) with or without behavioral
    information, depending on the 'target' parameter.

    The simulation can be conducted for a specified number of trials.

    Parameters
    ----------
    target : bool | False
        Indicates whether to include behavioral
        information (True) or not (False).
    n_trials : int | 1000
        Number of trials to simulate.
    triplet_character : str | null
        interaction character of the triplet of variables
        to generate. When triplet_character='synergy', a
        triplet of variable with negative O-information
        will be generated. If triplet_character='null'
        a triplet of variables with O-information close
        to zero will be generated. If
        triplet_character='redundancy' a triplet of
        variables with positive O-information
        will be generated.

    Returns
    -------
    Simulated data : numpy.ndarray
        A numpy array representing the simulated data.
        The shape of the generated data is n_trails, n_nodes
    """

    if not target:
        # withou target
        return sim_hoi(
            n_trials=n_trials,
            triplet_character=triplet_character,
        )

    elif target:
        # with target
        return sim_hoi_target(
            n_trials=n_trials,
            triplet_character=triplet_character,
        )


###############################################################################
###############################################################################
#                                   STATIC HOIs
###############################################################################
###############################################################################


def sim_hoi_target(
    n_trials=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) with behavioral information.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    triplet_character : str | "null"
        List of triplet characteristics.

    Returns
    -------
        simulated data without behavioral variable,
        behavioral variable : numpy.ndarray

    """

    mean_mvgauss = np.zeros(4)

    cov = cov_order_4(triplet_character)

    simulated_data = np.zeros((n_trials, 4))

    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_trials, check_valid="warn", tol=1e-8
    )

    return simulated_data[:, :3], simulated_data[:, 3]


def sim_hoi(
    n_trials=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) without behavioral information.

    Parameters
    ----------
    n_trials : int | 1000
        Number of trials to simulate.
    triplet_character : str | "null"
        List of triplet characteristics.

    Returns
    -------
    Simulated data : numpy.ndarray

    """

    # n_triplets = int(n_nodes/3)

    mean_mvgauss = np.zeros(3)
    cov = cov_order_3(triplet_character)

    simulated_data = np.zeros((n_trials, 3))

    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_trials, check_valid="warn", tol=1e-8
    )

    return simulated_data


###############################################################################
###############################################################################
#                        COVARIANCE WITH HOI, ORDER 3 & 4
###############################################################################
###############################################################################


def cov_order_3(character):
    """Compute the covariance matrix for three variables based
    on the given interaction character.

    Parameters
    ----------
    character : str
        'null', redundancy', or 'synergy' indicating the interaction
         character among the variables to simulate.

    Returns
    -------
    cov : numpy.ndarray
        covariance matrix for the three variables with the specified
        interaction pattern.
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
        theta_yz = 0.22

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three brain regions
        cov = m * m.T + theta

    elif character == "synergy":
        # We fix theta_yz in such a way that O(R1,R2,R3)<0
        theta_yz = -0.39

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three brain regions
        cov = m * m.T + theta

    return cov


def cov_order_4(character):
    """Calculate the covariance matrix for a given interaction
    character.

    Parameters
    ----------
    character : str
        The character specifying the kind of interactions character
        of the three variable and the target variable. It can
        be either 'redundancy' or 'synergy'.

    Returns
    -------
    cov_ : numpy.ndarray
    The covariance matrix for the specified interaction character.
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
