package ci.workshop.experiments.datasetgen;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.math.linearalgebra.Vector;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import ai.libs.jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import ci.workshop.experiments.storage.DatasetFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelineFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelinePerformanceStorage;
import ci.workshop.experiments.utils.Util;

public class DatasetGenerator {

	private List<Integer> pipelineIds;
	private int[] numTrainingPairs;
	private DatasetFeatureRepresentationMap datasetFeatures;
	private PipelineFeatureRepresentationMap pipelineFeatures;
	private PipelinePerformanceStorage pipelinePerformances;

	public DatasetGenerator(int[] numTrainingPairs, DatasetFeatureRepresentationMap datasetFeatures,
			PipelineFeatureRepresentationMap pipelineFeatures, PipelinePerformanceStorage pipelinePerformances) {
		this.pipelineIds = pipelinePerformances.getPipelineIds();
		this.numTrainingPairs = numTrainingPairs;
		this.datasetFeatures = datasetFeatures;
		this.pipelineFeatures = pipelineFeatures;
		this.pipelinePerformances = pipelinePerformances;
	}

	public int getRandomSeed(int splitnum, int numTrainingPairsIndex) {
		return splitnum * numTrainingPairs.length + numTrainingPairsIndex;
	}

	public static String getFileName(int splitnum, int numTrainingPairs) {
		return String.format("dyad_dataset_%d_%d.txt", splitnum, numTrainingPairs);
	}

	public DyadRankingDataset generateTrainingDataset(int splitnum, int numTrainingPairsIndex)
			throws IOException, URISyntaxException {
		// read train and test ids from json
		Pair<List<Integer>, List<Integer>> trainAndTestSplit = Util.getTrainingAndTestDatasetSplitsForSplitId(splitnum);
		List<Integer> trainInstances = trainAndTestSplit.getX();
		Random random = new Random(getRandomSeed(splitnum, numTrainingPairsIndex));

		List<IDyadRankingInstance> dyadRankingInstances = new ArrayList<>();

		// for every instance, sample numTrainingPairs as the train data
		trainInstances.forEach(datasetId -> {
			Vector instanceDatasetFeatures = new DenseDoubleVector(
					datasetFeatures.getFeatureRepresentationForDataset(datasetId));

			for (int i = 0; i < numTrainingPairs[numTrainingPairsIndex]; i++) {
				// sample first pipeline id and performance
				int pipeline1Id = pipelineIds.get(random.nextInt(pipelineIds.size()));
				double pipeline1Performance = pipelinePerformances
						.getPerformanceForPipelineWithIdOnDatasetWithId(pipeline1Id, datasetId);
				int pipeline2Id = pipeline1Id;
				double pipeline2Performance = pipeline1Performance;

				// sample 2nd pipeline that is different from the first
				while (pipeline1Performance == pipeline2Performance) {
					pipeline2Id = pipelineIds.get(random.nextInt(pipelineIds.size()));
					pipeline2Performance = pipelinePerformances
							.getPerformanceForPipelineWithIdOnDatasetWithId(pipeline2Id, datasetId);
				}

				// create dyads
				Vector instance1Features = new DenseDoubleVector(
						this.pipelineFeatures.getFeatureRepresentationForPipeline(pipeline1Id));
				Dyad dyad1 = new Dyad(instanceDatasetFeatures, instance1Features);
				Vector instance2Features = new DenseDoubleVector(
						this.pipelineFeatures.getFeatureRepresentationForPipeline(pipeline2Id));
				Dyad dyad2 = new Dyad(instanceDatasetFeatures, instance2Features);

				// add to ranking
				List<Dyad> dyads = new ArrayList<>();
				if (pipeline1Performance < pipeline2Performance) {
					dyads.add(dyad1);
					dyads.add(dyad2);
				} else {
					dyads.add(dyad2);
					dyads.add(dyad1);
				}

				DyadRankingInstance instance = new DyadRankingInstance(dyads);
				dyadRankingInstances.add(instance);
			}
		});

		return new DyadRankingDataset(dyadRankingInstances);
	}

	public static void main(String[] args) throws IOException, URISyntaxException {
		// make sure folder exists
		new File("datasets").mkdirs();
		
		// assume first args are host user pw db
		SQLAdapter sqlAdapter = new SQLAdapter(args[0], args[1], args[2], args[3]);
		int[] numTrainingPairs = { 100, 1000, 1900, 2750 };
		DatasetGenerator generator = new DatasetGenerator(numTrainingPairs,
				new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_mirror"),
				new PipelineFeatureRepresentationMap(sqlAdapter,
						"dyad_dataset_approach_5_performance_samples_with_SMO"),
				new PipelinePerformanceStorage(sqlAdapter, "pipeline_performance_5_classifiers_with_SMO"));

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < numTrainingPairs.length; j++) {
				DyadRankingDataset dataset = generator.generateTrainingDataset(i, j);
				OutputStream out = new FileOutputStream(new File("datasets/" + DatasetGenerator.getFileName(i, numTrainingPairs[j])));
				dataset.serialize(out);
			}
		}
	}
}