package ci.workshop.experiments.datasetgen;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.ml.core.exception.TrainingException;
import ai.libs.jaicore.ml.dyadranking.algorithm.featuretransform.FeatureTransformPLDyadRanker;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ci.workshop.experiments.storage.DatasetFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelineFeatureRepresentationMap;
import ci.workshop.experiments.storage.PipelinePerformanceStorage;

public class DatasetGeneratorTest {

	public void testGeneration() throws IOException, URISyntaxException, TrainingException {
		String host = "";
		String user = "";
		String pw = "";
		String db = "";
		SQLAdapter sqlAdapter = new SQLAdapter(host, user, pw, db);
		
		DatasetGenerator generator = new DatasetGenerator(new int[] { 10, 20, 30 },
				new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_mirror"),
				new PipelineFeatureRepresentationMap(sqlAdapter, "dyad_dataset_approach_5_performance_samples_full"),
				new PipelinePerformanceStorage(sqlAdapter, "pipeline_performance_5_classifiers"));
		System.out.println("Initialized generator");
		DyadRankingDataset dataset = generator.generateTrainingDataset(0, 0);

		System.out.println("Generated data");
		String fileName = "split_0_100_samples.data";
		try (FileOutputStream stream = new FileOutputStream(new File(fileName))) {
			dataset.serialize(stream);
		}
		System.out.println("wrote data");

		try (FileInputStream stream = new FileInputStream(new File(fileName))) {
			DyadRankingDataset data = new DyadRankingDataset();
			data.deserialize(stream);
			System.out.println("read data");
			
			// test if training is possible
			FeatureTransformPLDyadRanker ranker = new FeatureTransformPLDyadRanker();
			ranker.train(data);
		}
	}
}
