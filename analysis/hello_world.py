from datetime import datetime

# Get current date and time
now = datetime.now()

# Print date in yyyy-mm-dd format
print(now.strftime("%Y-%m-%d"))

# Print time in hh:mm format
print(now.strftime("%H:%M"))

# Print greeting
print("Hi Rexy!")
