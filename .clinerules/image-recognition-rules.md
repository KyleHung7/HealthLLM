# img_recognition.py

- Create blueprint to wrap all of the image recognition api route.
- Use LLM model to extract the value show in the image about the Blood pressure monitor and blood glucose meter.
- Get structured response from the api endpoint.

Model to use
- Gemini flash 2.0 - API KEY: env GEMINI_API_KEY

Routes
- bp_image
- bs_image

Blueprint Init
- Take auth.py as a reference