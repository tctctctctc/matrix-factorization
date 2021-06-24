# Matrix-Factorization

## Background  
  Pytorch implementation of matrix decomposition algorithm  
  [Zh Version](README.md)

## DateSet 
  uses the MovieLens dataset provided by GroupLens.  
  There are three different versions of the MovieLens. This chapter selects a medium-sized dataset.  
  The dataset contains more than 6000 users' 1 million ratings of more than 4000 movies.  
 
# Dataset Format  
	* ratings.dat: UserID::MovieID::Rating::Timestamp  
	
		- UserIDs range between 1 and 6040 
		- MovieIDs range between 1 and 3952
		- Ratings are made on a 5-star scale (whole-star ratings only)
		- Each user has at least 20 ratings

	* users.dat: UserID::Gender::Age::Occupation::Zip-code
	
		- Gender is denoted by a "M" for male and "F" for female
		- Age is chosen from the following ranges:
		
			*  1:  "Under 18"
			* 18:  "18-24"
			* 25:  "25-34"
			* 35:  "35-44"
			* 45:  "45-49"
			* 50:  "50-55"
			* 56:  "56+"
			
		- Occupation is chosen from the following choices:

			*  0:  "other" or not specified
			*  1:  "academic/educator"
			*  2:  "artist"
			*  3:  "clerical/admin"
			*  4:  "college/grad student"
			*  5:  "customer service"
			*  6:  "doctor/health care"
			*  7:  "executive/managerial"
			*  8:  "farmer"
			*  9:  "homemaker"
			* 10:  "K-12 student"
			* 11:  "lawyer"
			* 12:  "programmer"
			* 13:  "retired"
			* 14:  "sales/marketing"
			* 15:  "scientist"
			* 16:  "self-employed"
			* 17:  "technician/engineer"
			* 18:  "tradesman/craftsman"
			* 19:  "unemployed"
			* 20:  "writer"
	
	* movies.dat: MovieID::Title::Genres  
	
		- Genres are pipe-separated and are selected from the following genres:

			* Action
			* Adventure
			* Animation
			* Children's
			* Comedy
			* Crime
			* Documentary
			* Drama
			* Fantasy
			* Film-Noir
			* Horror
			* Musical
			* Mystery
			* Romance
			* Sci-Fi
			* Thriller
			* War
			* Western
  
## License
  [Apache-2.0 License](LICENSE)

