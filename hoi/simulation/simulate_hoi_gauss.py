import numpy as np

###############################################################################
###############################################################################
#                                 SWITCHER SIMULATIONS
###############################################################################
###############################################################################


def simulate_hoi_gauss(
    n_samples=1000,
    target=False,
    triplet_character="synergy",
):
    """Simulates High Order Interactions (HOIs) with or without target.

    This function can be used to simulate a triplet only using gaussian
    variables.

    Parameters
    ----------
    n_samples : int | 1000
        Number of samples to simulate.
    target : bool | False
        Indicates whether to include a target variable.
    triplet_character : {"redundancy", "synergy}
        Interaction character of the triplet of variables to generate. Choose
        either :

            * "redundancy" : a triplet of variables with positive O-information
              (about 0.3) will be generated
            * "synergy": a triplet of variables with negative O-information
              (about -0.8)

        In the case, target = True, triplet character refers to the information
        conveyed by the triplet with respect to the target variable.

    Returns
    -------
    data: np.ndarray
        Numpy array of simulated data of shape (n_samples, 3)
    target: np.ndarray
        Target variable of shape (n_samples,). This parameter is only returned
        when the input parameter target is set to True
    """
    assert triplet_character in ["redundancy", "synergy"]

    if not target:
        # without target
        return _sim_hoi(n_samples, triplet_character)

    elif target:
        # with target
        return _sim_hoi_target(n_samples, triplet_character)


###############################################################################
###############################################################################
#                                   STATIC HOIs
###############################################################################
###############################################################################


def _sim_hoi_target(n_samples, triplet_character):
    """Simulates High Order Interactions (HOIs) with target variable."""

    # Mean vector for multivariate Gaussian distribution with 4 variables
    mean_mvgauss = np.zeros(4)

    # Get the covariance matrix based on the triplet character
    cov = __cov_order_4(triplet_character)

    # Initialize an array to hold the simulated data
    simulated_data = np.zeros((n_samples, 4))

    # Generate the simulated data using a multivariate normal distribution
    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_samples, check_valid="warn", tol=1e-8
    )

    # Return the first three variables as the simulated data and the fourth as
    # # the target
    return simulated_data[:, :3], simulated_data[:, 3]


def _sim_hoi(n_samples, triplet_character):
    """Simulates High Order Interactions (HOIs) without target information."""

    # Mean vector for multivariate Gaussian distribution with 3 variables
    mean_mvgauss = np.zeros(3)

    # Get the covariance matrix based on the triplet character
    cov = __cov_order_3(triplet_character)

    # Initialize an array to hold the simulated data
    simulated_data = np.zeros((n_samples, 3))

    # Generate the simulated data using a multivariate normal distribution
    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_samples, check_valid="warn", tol=1e-8
    )

    # Return the simulated data
    return simulated_data


###############################################################################
###############################################################################
#                        COVARIANCE WITH HOI, ORDER 3 & 4
###############################################################################
###############################################################################


def __cov_order_3(character):
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
        Covariance matrix for the three variables with the specified
        interaction pattern.
    """

    # Define the standard deviations for the three variables
    lambx = np.sqrt(0.99)
    lamby = np.sqrt(0.7)
    lambz = np.sqrt(0.3)

    # Full factor matrix m
    m = np.array([lambx, lamby, lambz])[np.newaxis]

    if character == "redundancy":
        # We fix theta_yz in such a way that O(R1,R2,R3)>0
        theta_yz = 0.22

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three variables
        cov = m * m.T + theta

    elif character == "synergy":
        # We fix theta_yz in such a way that O(R1,R2,R3)<0
        theta_yz = -0.39

        # Noise covariances theta
        theta = np.diagflat(1 - m**2)
        theta += np.diagflat([0, theta_yz], 1) + np.diagflat([0, theta_yz], -1)

        # The covariance matrix for the three variables
        cov = m * m.T + theta

    return cov


def __cov_order_4(character):
    """Calculate the covariance matrix for a given interaction
    character.

    Parameters
    ----------
    character : str
        The character specifying the kind of interactions character
        of the three variables and the target variable. It can
        be either 'redundancy' or 'synergy'.

    Returns
    -------
    cov_ : numpy.ndarray
    The covariance matrix for the specified interaction character.
    """

    # Define the standard deviations for the four variables
    lambx = np.sqrt(0.99)
    lamby = np.sqrt(0.7)
    lambz = np.sqrt(0.3)
    lambs = np.sqrt(0.2)

    # Imposing the relationships with the target
    m = np.array([lambx, lamby, lambz, lambs])[np.newaxis]

    # Initialize the noise covariance matrix theta
    theta = np.diagflat(1 - m**2)

    if character == "redundancy":
        # We fix theta_zs in such a way that the variables are redundant
        theta_zs = 0.4

        # Update the noise covariance matrix theta
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )

        # Calculate the full covariance matrix
        cov_ = m * m.T + theta

    elif character == "synergy":
        # We fix theta_zs in such a way that the variables show synergy
        theta_zs = -0.75

        # Update the noise covariance matrix theta
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )

        # Calculate the full covariance matrix
        cov_ = m * m.T + theta

    return cov_
