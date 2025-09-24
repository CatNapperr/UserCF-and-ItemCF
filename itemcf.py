
import sys, random, math, os, csv
from operator import itemgetter


random.seed(0)


class ItemBasedCF():
    ''' TopN recommendation - ItemBasedCF '''
    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)


    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s succ' % filename, file=sys.stderr)


    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            if (random.random() < pivot):
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)


    def generate_dataset_kfold(self, filename, M=8, k=0, seed=0):
        '''
        K-fold split of rating data. Randomly assigns each interaction to one of M folds
        and uses fold k as test while others as train.
        '''
        self.trainset = {}
        self.testset = {}
        random.seed(seed)

        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            fold_id = random.randint(0, M - 1)
            if fold_id == k:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1
            else:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1

        print('k-fold split succ (M=%d, k=%d)' % (M, k), file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)


    def calc_movie_sim(self):
        ''' calculate movie similarity matrix '''
        print('counting movies number and popularity...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for movie in movies:
                # count item popularity 
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print('count movies number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2: continue
                    itemsim_mat.setdefault(m1,{})
                    itemsim_mat[m1].setdefault(m2,0)
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix 
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                        self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' % simfactor_count, file=sys.stderr)

        print('calculate movie similarity matrix(similarity factor) succ', file=sys.stderr)
        print('Total similarity factor number = %d' %simfactor_count, file=sys.stderr)


    def recommend(self, user):
        ''' Find K similar movies and recommend N movies. '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating_score in watched_movies.items():
            for related_movie, simi_score in sorted(self.movie_sim_mat[movie].items(),
                    key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += rating_score * simi_score
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


    def evaluate(self):
        ''' return precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall 
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count) if rec_count > 0 else 0.0
        recall = hit / (1.0 * test_count) if test_count > 0 else 0.0
        coverage = len(all_rec_movies) / (1.0 * self.movie_count) if self.movie_count > 0 else 0.0
        popularity = popular_sum / (1.0 * rec_count) if rec_count > 0 else 0.0

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' \
                % (precision, recall, coverage, popularity), file=sys.stderr)
        return precision, recall, coverage, popularity


    def cross_validate(self, filename, M=8, seed=0, csv_out='itemcf_cv_results.csv'):
        '''
        Run M-fold cross validation, save each fold's metrics and the average to CSV.
        '''
        results = []
        for k in range(M):
            print('\n==== Fold %d/%d ====' % (k + 1, M), file=sys.stderr)
            # prepare data for fold k
            self.generate_dataset_kfold(filename, M=M, k=k, seed=seed)
            # reset model state
            self.movie_sim_mat = {}
            self.movie_popular = {}
            self.movie_count = 0
            # train and evaluate
            self.calc_movie_sim()
            precision, recall, coverage, popularity = self.evaluate()
            results.append((k, precision, recall, coverage, popularity))

        # compute averages
        avg_precision = sum(r[1] for r in results) / M if M > 0 else 0.0
        avg_recall = sum(r[2] for r in results) / M if M > 0 else 0.0
        avg_coverage = sum(r[3] for r in results) / M if M > 0 else 0.0
        avg_popularity = sum(r[4] for r in results) / M if M > 0 else 0.0

        # save to CSV
        with open(csv_out, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['fold', 'precision', 'recall', 'coverage', 'popularity'])
            for k, p, r, c, pop in results:
                writer.writerow([k, p, r, c, pop])
            writer.writerow(['avg', avg_precision, avg_recall, avg_coverage, avg_popularity])

        print('\nCross-validation results saved to %s' % csv_out, file=sys.stderr)


if __name__ == '__main__':
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    itemcf = ItemBasedCF()
    # 确保K=20相似物品
    itemcf.n_sim_movie = 20
    # 运行8折交叉验证并保存结果
    itemcf.cross_validate(ratingfile, M=8, seed=0, csv_out='itemcf_cv_results.csv')
