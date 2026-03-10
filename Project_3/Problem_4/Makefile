pts1m = /home/oucspdn094/Project_3_Tests/test_data/Problem_4/points_1M.csv
pts4m = /home/oucspdn094/Project_3_Tests/test_data/Problem_4/points_4M.csv
pts16m = /home/oucspdn094/Project_3_Tests/test_data/Problem_4/points_16M.csv
centroid = /home/oucspdn094/Project_3_Tests/test_data/Problem_4/centroids.csv

all: make_clustering

# alter this
make_clustering: kmeans_clustering.c
	gcc -fopenmp -g -Wall -o kmeans_clustering kmeans_clustering.c -lm -std=c99
	
clean:
	$(RM) kmeans_clustering

# Below are commands to help you run your program easily.
# You can alter them to run multiple things at once.
run: make_clustering
	./kmeans_clustering 1000000 $(pts1m) 16 $(centroid) output_1m_1th.csv time_1m_1th.csv 1
	./kmeans_clustering 4000000 $(pts4m) 16 $(centroid) output_4m_1th.csv time_4m_1th.csv 1
	./kmeans_clustering 16000000 $(pts16m) 16 $(centroid) output_16m_1th.csv time_16m_1th.csv 1
	./kmeans_clustering 1000000 $(pts1m) 16 $(centroid) output_1m_4th.csv time_1m_4th.csv 4
	./kmeans_clustering 4000000 $(pts4m) 16 $(centroid) output_4m_4th.csv time_4m_4th.csv 4
	./kmeans_clustering 16000000 $(pts16m) 16 $(centroid) output_16m_4th.csv time_16m_4th.csv 4
	./kmeans_clustering 1000000 $(pts1m) 16 $(centroid) output_1m_16th.csv time_1m_16th.csv 16
	./kmeans_clustering 4000000 $(pts4m) 16 $(centroid) output_4m_16th.csv time_4m_16th.csv 16
	./kmeans_clustering 16000000 $(pts16m) 16 $(centroid) output_16m_16th.csv time_16m_16th.csv 16