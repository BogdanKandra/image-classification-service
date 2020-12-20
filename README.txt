Regarding the model file:
-> Currently, the model will be copied to the image file system if it exists and the `create_model.py` script will create it directly in the image if it did not exist.
-> This would not be acceptable in the case of a model which has to be trained, but this could be bypassed by storing the model in a cloud location, for the script to download (which was not the case in the present task)

Building the application:
-> docker-compose build classification
-> docker-compose up classification
    The web application is now running on http://localhost:8060

Calling the web-service:
-> From the web browser, visit http://localhost:8060/swagger for viewing the Swagger UI and testing the endpoint
-> From the terminal, use `curl -F image=@fileName http://localhost:8060/classifyImage` for classifying the specified file using the VGG 16 classifier