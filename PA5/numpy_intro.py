import numpy as np

kobe = [[18, 7.6], [19, 15.4], [20, 19.9], [21, 22.5], [22, 28.5], [23, 25.2],
        [24, 30], [25, 24], [26, 27.6], [27, 35.4], [28, 31.6], [29, 28.3],
        [30, 26.8], [31, 27], [32, 25.3], [33, 27.9], [34, 27.4], [35, 13.8],
        [36, 22.3], [37, 17.6]]

# Part2: TODO- Make kobe into numpy array (kobe_np) and getting dimensions
kobe_np = np.array(kobe)
num_rows = kobe_np.shape[0]
num_cols = kobe_np.shape[1]

# Part3a: TODO- Make transpose of kobe_np (kobe_transpose)
kobe_transpose = kobe_np.T

# Part3b: TODO- Ones
ones = np.ones(num_rows)

# Part3c: TODO- Getting value ([18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37.])
A = kobe_transpose[0]
y = kobe_transpose[1]

# Part3d: TODO- Use column_stack
x = np.column_stack((A, ones))

# Part3e: TODO- Use matrix multiplication
x_prod = np.matmul(x.T, x)

# Part3f: TODO- Find inverse
x_prod_inv = np.linalg.inv(x_prod)

# Part 4:
theta = np.matmul(x_prod_inv, np.matmul(x.T, y))
print("x_prod_inv:\n", x_prod_inv, "\ntheta0:\n", theta[0], "\ntheta1:\n", theta[1])
