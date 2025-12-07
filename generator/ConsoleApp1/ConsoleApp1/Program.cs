namespace ConsoleApp1
{
    using System;
    using System.IO;
    using System.Security.Cryptography;

    public static class Program
    {
        // SIFT descriptor dimensionality
        private const int SiftDim = 128;

        public static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("Usage: dotnet run -- <numVectors> [outputPath]");
                return 1;
            }

            if (!int.TryParse(args[0], out int numVectors) || numVectors < 0)
            {
                Console.Error.WriteLine("Error: <numVectors> must be a non-negative integer.");
                return 1;
            }

            string outputPath = args.Length >= 2
                ? args[1]
                : $"sift_{numVectors}.bvecs";

            try
            {
                CreateSiftBvecs(outputPath, numVectors, SiftDim);
                Console.WriteLine($"Created '{outputPath}' with {numVectors} SIFT vectors (dim={SiftDim}).");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Creates a .bvecs-like file:
        /// For each vector: int32 dimension followed by 'dimension' bytes.
        /// </summary>
        public static void CreateSiftBvecs(string path, int count, int dimension = SiftDim)
        {
            // Ensure directory exists if user passed a folder-based path
            var dir = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrWhiteSpace(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            // Use cryptographically strong RNG to avoid predictable data.
            // If you want reproducibility, replace with Random(seed).
            using var rng = RandomNumberGenerator.Create();
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
            using var bw = new BinaryWriter(fs);

            byte[] buffer = new byte[dimension];

            for (int i = 0; i < count; i++)
            {
                rng.GetBytes(buffer);

                // Write dimension (little-endian int32 by default in BinaryWriter)
                bw.Write(dimension);

                // Write descriptor bytes
                bw.Write(buffer);
            }

            bw.Flush();
        }
    }

}
