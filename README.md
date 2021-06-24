# Matrix-Factorization

## 简介  
  矩阵分解算法pytorch实现
      
## 数据集
  采用GroupLens提供的MovieLens数据集  
  MovieLens数据集有3个不同版本，本章选用中等大小的数据集  
  该数据集包含6000多名用户对4000多部电影的100万条评分
  
# 数据集格式  
	* ratings.dat: UserID::MovieID::Rating::Timestamp  
	
		- UserID:用户ID范围从1到6040  
		- MovieID:电影ID范围从1到3952  
		- Ratings:评分有1到5的5个等级
		- 每个用户最少有20条评分数据

	* users.dat: UserID::Gender::Age::Occupation::Zip-code
	
		- Gender: "M"代表男，"F"代表女  
		- Age: 年龄从下面的范围中选择
		
			*  1:  "Under 18"
			* 18:  "18-24"
			* 25:  "25-34"
			* 35:  "35-44"
			* 45:  "45-49"
			* 50:  "50-55"
			* 56:  "56+"
			
		- Occupation: 职业包括下面的类别
		
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
	
		- Genres:电影类别包括以下的类别  
		
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

