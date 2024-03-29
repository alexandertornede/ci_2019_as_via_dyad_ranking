package ci.workshop.experiments.evaluator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.tsc.distances.EuclideanDistance;
import ci.workshop.experiments.loss.KendallsTauBasedOnApache;
import ci.workshop.experiments.loss.KendallsTauBasedOnApacheAndRanks;
import ci.workshop.experiments.loss.Metric;
import ci.workshop.experiments.loss.PerformanceDifferenceOfAverageOnTopK;
import ci.workshop.experiments.loss.PerformanceDifferenceOfBestOnTopK;
import ci.workshop.experiments.rankers.AveragePerformanceRanker;
import ci.workshop.experiments.rankers.AverageRankBasedRanker;
import ci.workshop.experiments.rankers.DyadRankingBasedRanker;
import ci.workshop.experiments.rankers.IdBasedRanker;
import ci.workshop.experiments.rankers.KnnRanker;
import ci.workshop.experiments.storage.DatasetFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelineFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelinePerformanceStorage;
import ci.workshop.experiments.utils.Util;

public class ExperimentRunner {

	private static String databaseName = "conference_ci-workshop2019"; // "conference_ci-workshop2019"
	private static String tableName = "all_results_rank_all_with_values_with_smo";

	private static String pathToStoredRankingModels = "rankersMCCV";

	public static void main(String[] args) throws Exception {

		// int[] numberOfPairwiseSamplesPerDatasetSizes = { 100, 1000, 1900, 2750 };
		int[] numberOfPairwiseSamplesPerDatasetSizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2250, 2500, 2750 };

		int experimentNumber = 0;
		for (int datasetTestSplitId = 0; datasetTestSplitId < 10; datasetTestSplitId++) {
			Pair<List<Integer>, List<Integer>> trainTestSplit = Util.getTrainingAndTestDatasetSplitsForSplitId(datasetTestSplitId);

			List<Integer> trainindDatasetIds = trainTestSplit.getX();
			List<Integer> testDatasetIds = trainTestSplit.getY();

			SQLAdapter sqlAdapter = new SQLAdapter("isys-db.cs.upb.de:3306", "user", "password", databaseName, true);

			PipelineFeatureRepresentationMap pipelineFeatureMap = new PipelineFeatureRepresentationMap(sqlAdapter, "dyad_dataset_approach_5_performance_samples_with_SMO"); // "dyad_dataset_approach_5_performance_samples_with_SMO"
			DatasetFeatureRepresentationMap datasetFeatureMap = new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_mirror"); // "dataset_metafeatures_mirror"
			PipelinePerformanceStorage pipelinePerformanceStorage = new PipelinePerformanceStorage(sqlAdapter, "pipeline_performance_5_classifiers_with_SMO"); // "pipeline_performance_5_classifiers_with_SMO"

			AveragePerformanceRanker averageRankRanker = new AveragePerformanceRanker(pipelinePerformanceStorage);
			KnnRanker onennRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 1);
			KnnRanker twonnRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 2);
			AverageRankBasedRanker averageRankBasedRanker = new AverageRankBasedRanker(pipelinePerformanceStorage);

			List<IdBasedRanker> rankers = new ArrayList<>(Arrays.asList(averageRankRanker, onennRanker, twonnRanker, averageRankBasedRanker));
			// List<IdBasedRanker> rankers = new ArrayList<>(Arrays.asList(averageRankRanker, onennRanker, averageRankBasedRanker));

			for (int numberOfPairwiseSamplesPerDataset : numberOfPairwiseSamplesPerDatasetSizes) {
				PLNetDyadRanker dyadRanker = getDyadRankerForNumberOfPairwiseSamples(numberOfPairwiseSamplesPerDataset, datasetTestSplitId);
				DyadRankingBasedRanker dyadRankingBasedRanker = new DyadRankingBasedRanker(numberOfPairwiseSamplesPerDataset, dyadRanker, pipelineFeatureMap, datasetFeatureMap);
				rankers.add(dyadRankingBasedRanker);
			}

			List<Metric> metrics = Arrays.asList(new KendallsTauBasedOnApache(), new KendallsTauBasedOnApacheAndRanks(), new PerformanceDifferenceOfAverageOnTopK(3), new PerformanceDifferenceOfBestOnTopK(3),
					new PerformanceDifferenceOfAverageOnTopK(5), new PerformanceDifferenceOfBestOnTopK(5));
			int numberOfTestPipelineSets = 1; // 1, since we rank all

			for (IdBasedRanker ranker : rankers) {
				Experiment experiment = new Experiment(pipelinePerformanceStorage, sqlAdapter, databaseName, tableName);
				experiment.runExperiment(datasetTestSplitId, trainindDatasetIds, testDatasetIds, ranker, metrics, numberOfTestPipelineSets);
				System.out.println("Experiment " + experimentNumber + " / " + (10 * rankers.size()) + " done.");
				experimentNumber++;
			}

			sqlAdapter.close();
		}

	}

	private static PLNetDyadRanker getDyadRankerForNumberOfPairwiseSamples(int numberOfPairwiseSamplesPerDataset, int datasetTestSplitId) throws IOException {
		PLNetDyadRanker dyadRanker = new PLNetDyadRanker();
		String filePath = pathToStoredRankingModels + "/ranker_" + numberOfPairwiseSamplesPerDataset + "_" + datasetTestSplitId + ".zip";
		// System.out.println("Loading: " + filePath);
		dyadRanker.loadModelFromFile(filePath);
		return dyadRanker;
	}

}
