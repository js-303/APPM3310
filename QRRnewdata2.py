import numpy as np
import matplotlib.pyplot as plt

#Prep dictionaries and lists
monthly = {}
yearly= {}
month = []
b = []

#Insert data into into list and create A with periodic fit
with open("C:\\Users\\jamil\\Documents\\APPM3310\\stressdata.txt", "r") as file: #insert .txt file path into ("...")
    stressdata = [float(line.rstrip()) for line in file]
file.close()

with open("C:\\Users\\jamil\\Documents\\APPM3310\\straindata.txt", "r") as file: #insert .txt file path into ("...")
    straindata = [float(line.rstrip()) for line in file]
file.close()

eps = np.array(straindata)
A = np.vstack([np.ones(len(eps)), eps]).T
b = np.array(stressdata)

# QRR core functions
def QRR(A):
    """
    Recursive QR decomposition algorithm (QRR).
    """
    n, m = A.shape

    #Base case: m = 1 (single column)
    if m == 1:
        #Compute conventional Householder transformation
        a = np.linalg.norm(A, 2)
        e1 = np.zeros_like(A)
        e1[0, 0] = 1
        
        #Handle the sign to avoid cancellation
        if A[0,0] != 0:
            sign_val = np.sign(A[0,0])
        else:
            sign_val = 1
        #sign_val = np.sign(A[0, 0]) if A[0, 0] != 0 else 1
        v = A + sign_val * a * e1
        
        #Normalize v to get unit vector
        u = v / np.linalg.norm(v, 2)
        
        #W has unit 2-norm columns n×1
        W = u
        
        #Y has 2-norm equal to 2 rows - shape is 1×n
        Y = 2 * u.T  # Transpose to make Y 1×n
        
        #Create m×m R matrix 1x1
        R = np.array([[-sign_val * a]])
        
        #Clean R, turn very small numbers to 0
        R[np.abs(R) < 1e-13] = 0

        return R, W, Y
    
    #Recursive case
    #Create floor and ceiling variables
    floor_m_2 = m // 2
    ceil_m_2 = m - floor_m_2
    
    #Compute QR decomposition of left half
    left_half = A[:, :floor_m_2]
    R_L, W_L, Y_L = QRR(left_half)
    
    #Update right half of A
    A_right = A[:, floor_m_2:].copy()
    Y_L_A_right = Y_L @ A_right
    W_L_Y_L_A_right = W_L @ Y_L_A_right
    A_right = A_right - W_L_Y_L_A_right
    
    #Compute QR decomposition of bottom-right block
    A_right_bottom = A_right[floor_m_2:, :]
    R_R, W_R, Y_R = QRR(A_right_bottom)
    
    #Construct X matrix
    zeros_top = np.zeros((floor_m_2, floor_m_2))
    W_L_bottom = W_L[floor_m_2:, :]
    Y_R_W_L_bottom = Y_R @ W_L_bottom
    X_bottom = W_R @ Y_R_W_L_bottom
    X_diff = np.vstack([zeros_top, X_bottom])
    X = W_L - X_diff
    
    #Construct R matrix m×m 
    R = np.zeros((m, m))
    R[:floor_m_2, :floor_m_2] = R_L
    R[:floor_m_2, floor_m_2:] = A_right[:floor_m_2, :]
    R[floor_m_2:, floor_m_2:] = R_R
    #Clean R
    R[np.abs(R) < 1e-13] = 0.0

    #Construct W matrix n×m
    W_right = np.vstack([np.zeros((floor_m_2, ceil_m_2)), W_R])
    W = np.hstack([X, W_right])
    
    #Construct Y matrix m×n
    Y_top = Y_L
    Y_R_padded = np.zeros((ceil_m_2, n))
    Y_R_padded[:, floor_m_2:] = Y_R
    Y = np.vstack([Y_top, Y_R_padded])

    return R, W, Y

R, W, Y = QRR(A)

def construct_Q(W, Y):
    """
    Construct Q matrix from W and Y where Q = I - WY.
    """
    n = W.shape[0]
    Q_T = np.eye(n) - W@Y
    Q =Q_T.T

    return Q

Q = construct_Q(W, Y)

def least_squares_qrr(A, Q, R, b):
    """Solve least squares problem using QRR algorithm"""
    m = A.shape[1]

    #Solve least squares: x = R^-1 * Q^T * b
    Qb = Q.T @ b

    #Back-substitution to solve R*x = Q^T*b with BLAS
    x = np.linalg.solve(R, Qb[:m])
    return x

#QRR implementation
QR = least_squares_qrr(A, Q, R, b)
print(QR)
n,m = A.shape
R_pad = np.vstack([R,np.zeros((n-m,m))])
print("R_pad shape", R_pad)
print("A:",A)
print("Q@R", Q@R_pad)
print("Error:", np.linalg.norm(A-Q@R_pad,2))

f = lambda eps: QR[0]+QR[1]*eps
tt = np.linspace(0, 0.005, 10000)
ax = plt.axes()
plt.plot(tt, f(tt), label='linear fit')
for i in range(len(b)):
   plt.plot(eps[i], b[i], marker="o", c='black')
plt.title("Title")
plt.ylabel("stress")
plt.xlabel("strain")
plt.legend()
plt.show()