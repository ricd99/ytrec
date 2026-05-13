# videos while you eat.
[run the app here!](https://videos-while-you-eat-825100164825.us-west1.run.app/ui/)

I found myself refreshing my YouTube recommendations for 10+ minutes one night when I ate alone. The recommendations were based on my recent watch history, but I wanted to watch a mystery/crime video essay (something I hadn't watched in weeks) in the style of LEMMiNO (great channel). YouTube's recommendations can be too skewed towards recent activity, and YouTube lacks a channel-based recommendation function, so I built a web app to solve these two problems.

## Technologies
This is a non-exhaustive list that covers the core technologies used for this project. 

- Python
  - pandas
  - scikit-learn
  - sentence-transformers
  - FastAPI
- Hugging Face
- Google Cloud Run

## My Process

in progress, please excuse the word salad.

This repo is split into two parts. The first part develops the nearest neighbor model that identifies similar channels. This contains the ETL pipeline + model training + uploading the model to Hugging Face. The second part is the web app that handles user queries. This includes receiving a channel name as input, turning the channel name into data for the model, and retrieving and displaying the prediction. The scripts that handle model training are located in the scripts/ directory, and the ui is located at app/main.py. Both parts rely on code located in the src/ directory. Repository organization improvements are possible; it is something I'm actively thinking about. Suggestions are welcome!

- Model training
  - First, I wrote a script using the YouTube API to collect channel data by simply searching. My searches started broad ("video essay"), but I think it is more effective to search at a more specific level to get the channels that tell the most interesting stories ("biggest choke in sports history", "jazz music's biggest controversy"). Two API calls are needed: one to collect channel data, the second to collect keywords from recently uploaded videos. I also have filters to avoid collecting inactive/truly beginner-level channels.
- The next stage is the ETL pipeline to clean and transform data for the model. I was planning to use AWS to deploy my project (I did not end up doing this, as you'll read later on), so I used the S3 object storage and RDS database in AWS to store my data. I dump the raw YouTube API data into S3, clean the data and dump it into the first table in my database, and then engineer features (in this case, it was simply combining all text columns into one big column) and dump it into the second table. The data is ready to be trained on!
- I first experimented with CountVectorizer from scikit-learn, which is simply a Bag of Words (BoW) model. This did not give good results, so I switched gears and tried a sentence embedding model. This also did not give good results! The key feature engineering step that dramatically improved results was combining all text columns into one giant column. This made sense for the BoW model, as it simply counts word occurrences, and one giant column gives the most accurate word count representation. I tested this new engineered dataset with sentence embeddings, and the results were even better. I'm not sure exactly why, but perhaps it's because this gives the model the most information possible in one window, which results in a better understanding compared to when the text was spread out over multiple shorter columns. Finally, if you're wondering how I evaluated the results, I simply tested a bunch of common channels, searched up the recommended one, and judged their similarity manually. If there is a meaningful metric to do this automatically, please let me know!
- Finally, I uploaded the model to Hugging Face, where my web app will call it.
  
- Web App
  - I used Gradio to develop a simple UI and mounted it using FastAPI (/ui)
  - I also exposed a REST API endpoint using FastAPI that provides the same service. (/predict)

There was not much difficulty here, partly because I kept it simple on purpose, and partly because Gradio and FastAPI are pleasant to set up and use!

## Results and next steps
I am satisfied with the results. Searches for video essay channels on a variety of topics (history, movies, sports...) have reasonable results. I believe the biggest limitation is not the model, but rather the quality of the dataset. I am thinking about more effective ways to find high-quality channels that may not have much visibility and weed out AI-slop channels.

- The lookup table that turns the model's predictions into  YouTube channel names is uploaded to Hugging Face alongside the model. Instead, I'd like to retrieve the channel names by referring to the database, so an extra lookup table artifact does not need to be made at training time.
- Introduce a "popular" vs "hidden gem" setting that allows users to choose whether they want popular or unknown channels to be recommended.
- Experiment using FAISS instead of scikit-learn's knn. I'm curious what the differences will be, or even if there will be differences?
- Freshen up the UI. I barely deviated from a Gradio UI template. Some quick updates I plan to implement are eliminating unnecessary buttons/adding a cancel request button, and formatting the output more nicely (as it just appears in a textbox right now). 
- Ultimately, I think this app is more effective as a browser extension.

Especially for the last point (but for others as well), I welcome others to reach out and help me!


**The first query may take around 20s, but every query after that is usually under 3s. This is because the app completely shuts down if no requests are coming in to save my Google credits!
