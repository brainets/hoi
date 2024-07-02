import numpy as np

###############################################################################
###############################################################################
#                                 SWITCHER SIMULATIONS
###############################################################################
###############################################################################


def simulate_hois_gauss(
    target=False,
    n_samples=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) with or without target
    variable, depending on the 'target' parameter.

    The parameter 'n_samples' allows to choose the length of the simulated
    data.

    Parameters
    ----------
    target : bool | False
        Indicates whether to include target
        variable (True) or not (False).
    n_samples : int | 1000
        Number of samples to simulate.
    triplet_character : str | null
        Interaction character of the triplet of variables
        to generate. When triplet_character='synergy', a
        triplet of variables with negative O-information (about -0.8)
        will be generated. If triplet_character='null'
        a triplet of variables with O-information close
        to zero will be generated. If
        triplet_character='redundancy' a triplet of
        variables with positive O-information (about 0.3)
        will be generated. In the case, target = True, triplet
        character refers to the information conveyed by the triplet
        with respect to the target variable.

    Returns
    -------
    Simulated data, target variable (if target) : numpy.ndarray
        A numpy array representing the simulated data.
        An array representing the target variable, if target=True
        The shape of the generated data is n_samples, n_variables
    """

    if not target:
        # without target
        return sim_hoi(
            n_samples=n_samples,
            triplet_character=triplet_character,
        )

    elif target:
        # with target
        return sim_hoi_target(
            n_samples=n_samples,
            triplet_character=triplet_character,
        )


###############################################################################
###############################################################################
#                                   STATIC HOIs
###############################################################################
###############################################################################


def sim_hoi_target(
    n_samples=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) with target variable.

    Parameters
    ----------
    n_samples : int | 1000
        Number of samples to simulate.
    triplet_character : str | "null"
        List of triplet characteristics.

    Returns
    -------
        Simulated data without target variable,
        target variable : numpy.ndarray

    """

    # Mean vector for multivariate Gaussian distribution with 4 variables
    mean_mvgauss = np.zeros(4)

    # Get the covariance matrix based on the triplet character
    cov = cov_order_4(triplet_character)

    # Initialize an array to hold the simulated data
    simulated_data = np.zeros((n_samples, 4))

    # Generate the simulated data using a multivariate normal distribution
    simulated_data = np.random.multivariate_normal(
        mean_mvgauss, cov, size=n_samples, check_valid="warn", tol=1e-8
    )

    # Return the first three variables as the simulated data and the fourth as
    # # the target
    return simulated_data[:, :3], simulated_data[:, 3]


def sim_hoi(
    n_samples=1000,
    triplet_character="null",
):
    """Simulates High Order Interactions (HOIs) without target information.

    Parameters
    ----------
    n_samples : int | 1000
        Number of samples to simulate.
    triplet_character : str | "null"
        List of triplet characteristics.

    Returns
    -------
    Simulated data : numpy.ndarray

    """

    # Mean vector for multivariate Gaussian distribution with 3 variables
    mean_mvgauss = np.zeros(3)

    # Get the covariance matrix based on the triplet character
    cov = cov_order_3(triplet_character)

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
        Covariance matrix for the three variables with the specified
        interaction pattern.
    """

    # Define the standard deviations for the three variables
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

        # The covariance matrix for the three variables
        cov = m * m.T + theta

    elif character == "redundancy":
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


def cov_order_4(character):
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
        theta_zs = 0.25

        # Update the noise covariance matrix theta
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )

        # Calculate the full covariance matrix
        cov_ = m * m.T + theta

    if character == "synergy":
        # We fix theta_zs in such a way that the variables show synergy
        theta_zs = -0.52

        # Update the noise covariance matrix theta
        theta += np.diagflat([0, 0, theta_zs], 1) + np.diagflat(
            [0, 0, theta_zs], -1
        )

        # Calculate the full covariance matrix
        cov_ = m * m.T + theta

    return cov_


if __name__ == "__main__":
    from hoi.metrics import Oinfo
    from hoi.utils import get_nbest_mult

    # simulate HOIs
    x = simulate_hois_gauss()

    # initialize an array to hold the O-information values
    oi = np.zeros((3, 4))  # x

    # create an Oinfo model
    model = Oinfo(x=oi)

    # fit the model to the simulated data
    hoi = model.fit(x)

    # print the results and check that they correspond to the ground truth
    df = get_nbest_mult(hoi, model=model)
