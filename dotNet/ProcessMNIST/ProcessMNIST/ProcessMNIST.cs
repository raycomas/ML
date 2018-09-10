using System;
using System.IO;
using System.Linq;

namespace ProcessMNIST
{
    // Create randomly selected training and test sets from the MNIST files

    class ProcessMNIST
    {
        // Names of the MNIST data files
        private const string TrainingDataFile = "train-images.idx3-ubyte";
        private const string TrainingLabelFile = "train-labels.idx1-ubyte";
        private const string TestDataFile = "t10k-images.idx3-ubyte";
        private const string TestLabelFile = "t10k-labels.idx1-ubyte";

        // Name of the data files we're going to create
        private const string TrainingOutputFile = "training-data.txt";
        private const string TestOutputFile = "test-data.txt";

        private const int TRAIN_DATA_MAX_SIZE = 60000;  // Number of images in MNIST training set
        private const int TEST_DATA_MAX_SIZE = 10000;   // Number of images in MNIST test set

        private const int TRAIN_DATA_SIZE = 12000;  // Number of images to extract into our training set
        private const int TEST_DATA_SIZE = 4000;    // Number of images to extract into our test set

        static void Main(string[] args)
        {
            string trainingDataPath = String.Empty;
            string trainingLabelPath = String.Empty;
            string testDataPath = String.Empty;
            string testLabelPath = String.Empty;
            string trainingOutputPath = String.Empty;
            string testOutputPath = String.Empty;

            int train_data_size = TRAIN_DATA_SIZE;
            int test_data_size = TEST_DATA_SIZE;

            try
            {
                if ((args.Length != 1 && args.Length != 3))
                    throw new ApplicationException("Incorrect number of arguments.");

                // Command line param is the path to the MNIST files. Output will be placed there, as well
                if (!Directory.Exists(args[0]))
                    throw new ApplicationException($"Directory not found: {args[0]}");

                if (args.Length == 3)
                {
                    if (Int32.TryParse(args[1], out train_data_size))
                        train_data_size = Math.Abs(1000 * train_data_size);
                    else
                        train_data_size = TRAIN_DATA_SIZE;

                    if (train_data_size > TRAIN_DATA_MAX_SIZE)
                        throw new ApplicationException($"Training set size can be at most {TRAIN_DATA_MAX_SIZE}");

                    if (Int32.TryParse(args[2], out test_data_size))
                        test_data_size = Math.Abs(1000 * test_data_size);
                    else
                        test_data_size = TEST_DATA_SIZE;

                    if (test_data_size > TEST_DATA_MAX_SIZE)
                        throw new ApplicationException($"Test set size can be at most {TEST_DATA_MAX_SIZE}");
                }


                trainingDataPath = Path.Combine(args[0], TrainingDataFile);
                if (!File.Exists(trainingDataPath))
                    throw new ApplicationException($"Cannot find file {trainingDataPath}");

                trainingLabelPath = Path.Combine(args[0], TrainingLabelFile);
                if (!File.Exists(trainingLabelPath))
                    throw new ApplicationException($"Cannot find file {trainingLabelPath}");

                testDataPath = Path.Combine(args[0], TestDataFile);
                if (!File.Exists(testDataPath))
                    throw new ApplicationException($"Cannot find file {testDataPath}");

                testLabelPath = Path.Combine(args[0], TestLabelFile);
                if (!File.Exists(testLabelPath))
                    throw new ApplicationException($"Cannot find file {testLabelPath}");

                trainingOutputPath = Path.Combine(args[0], TrainingOutputFile);
                if (File.Exists(trainingOutputPath))
                    File.Delete(trainingOutputPath);

                testOutputPath = Path.Combine(args[0], TestOutputFile);
                if (File.Exists(testOutputPath))
                    File.Delete(testOutputPath);

                if (!String.IsNullOrEmpty(trainingDataPath) && !String.IsNullOrEmpty(trainingLabelPath))
                {
                    Console.WriteLine("Processing training data ...");
                    ProcessDataSet(trainingDataPath, trainingLabelPath, trainingOutputPath,
                                            RandomElements(TRAIN_DATA_MAX_SIZE, TRAIN_DATA_SIZE));
                    Console.WriteLine("...Done");
                }

                if (!String.IsNullOrEmpty(testDataPath) && !String.IsNullOrEmpty(testLabelPath))
                {
                    Console.WriteLine("Processing test data ...");
                    ProcessDataSet(testDataPath, testLabelPath, testOutputPath,
                                            RandomElements(TEST_DATA_MAX_SIZE, TEST_DATA_SIZE));
                    Console.WriteLine("...Done");
                }
            }
            catch (ApplicationException ex)
            {
                Console.WriteLine(ex.Message);
                Show_Usage();
            }
        }

        private static void ProcessDataSet(string imageDataPath, string labelPath, string outputPath, int[] sample_indices = null)
        {
            byte[] labels = ReadLabels(labelPath);
            Console.WriteLine($"\tProcessing {labels.Length} images");

            using (BinaryReader imageReader = OpenImageDataReader(imageDataPath))
            using (StreamWriter outputWriter = new StreamWriter(outputPath))
            {
                for (int ix = 0, written = 0; ix < labels.Length; ++ix)
                {
                    if (ix % 1000 == 0)
                        Console.WriteLine($"\t... {ix} images processed");

                    byte[] image = imageReader.ReadBytes(28 * 28);
                    if (sample_indices == null || ix == sample_indices[written])
                    {
                        WriteImage(outputWriter, labels[ix], image);
                        if (sample_indices != null && ++written == sample_indices.Length)
                            break;
                    }
                }

                outputWriter.Close();
                imageReader.Close();
            }
        }

        private static void WriteImage(StreamWriter outputFile,byte label, byte[] image)
        {
            outputFile.Write(label);
            for (int iy = 0; iy < image.Length; ++iy)
                outputFile.Write($",{(image[iy] / 255.0).ToString("G6")}");
            outputFile.WriteLine();
        }

        private static byte[] ReadLabels(string filePath)
        {
            byte[] rawBytes = File.ReadAllBytes(filePath);
            byte[] result = new byte[rawBytes.Length - 8];
            Array.Copy(rawBytes, 8, result, 0, result.Length);
            return result;
        }

        private static BinaryReader OpenImageDataReader(string path)
        {
            BinaryReader rdr = new BinaryReader(File.Open(path, FileMode.Open));
            byte[] magicNumber = rdr.ReadBytes(4);
            if (!magicNumber.SequenceEqual(new byte[] { 0, 0, 0x08, 0x03 }))
                throw new ApplicationException($"Incorrect magic number in training image file {path}");

            rdr.ReadBytes(12); // Skip header info
            return rdr;
        }

        private static int[] RandomElements(int size, int sample)
        {
            int[] items = Enumerable.Range(0, size).ToArray();

            // Generate a random permutaion of "size" elements
            Random rand = new Random();
            for (int ix = 0; ix < size; ++ix)
            {
                int r = ix + rand.Next(size - ix);
                int temp = items[r];
                items[r] = items[ix];
                items[ix] = temp;
            }

            // Grab the first "sample" elements and sort them
            if (sample >= size)
                return items.OrderBy(i => i).ToArray();

            int[] result = new int[sample];
            Array.Copy(items, 0, result, 0, sample);

            return result.OrderBy(i => i).ToArray();
        }

        private static void Show_Usage()
        {
            Console.WriteLine("Specify the path to MNIST files, and size (in 1000s) of the training and test sets");
            Console.WriteLine("For example:\n");
            Console.WriteLine("\tProcessMNIST \"C:\\Some Directory\\MNIST\" 10 2\n");
            Console.WriteLine("Creates a training set with 10000 images, and a test set with 2000 images");
        }
    }
}
