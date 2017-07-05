def loadMovieList():
    fid = open("movie_ids.txt")
    n1 = 1682
    movieList = [[] for i in range(n1)]
    for i in range(0,n1):
        line = fid.readline()
        indx = line.index(" ")
        idx = line[0:indx]
        movieName = line[indx:]
        movieList[i].append(movieName.strip())
    fid.close()
    return movieList

