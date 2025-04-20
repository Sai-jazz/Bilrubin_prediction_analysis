

x = []
y = []

# Ask the user how many inputs they want to give
x_inputs = int(input("no. of  x co-ordinates: "))
# Collect inputs and store them in the list
for i in range(x_inputs):
    user_input_x =float(input(f" {i+1}. x co-ordinate: "))
    x.append(user_input_x)


for i in range(x_inputs):
    user_input_y = float(input(f"{i+1}. y co-ordinate: "))
    y.append(user_input_y)
