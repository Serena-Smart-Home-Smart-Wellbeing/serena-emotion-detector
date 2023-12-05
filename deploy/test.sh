# Start the Flask server in the background
python main.py &

# Save the Flask server's PID
FLASK_PID=$!

# Send the images
for image in test/images/*; do
  echo $image  
  curl -X POST -F "file=@$image" http://127.0.0.1:5000
done

# Stop the Flask server
kill $FLASK_PID