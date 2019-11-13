mongoimport -d yelp -c checkin  dataset\checkin.json
mongoimport -d yelp -c business dataset\business.json
mongoimport -d yelp -c tip      dataset\tip.json
mongoimport -d yelp -c user     dataset\user.json
mongoimport -d yelp -c review   dataset\review.json