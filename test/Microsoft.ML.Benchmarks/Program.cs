// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Benchmarks
{
    class Program
    {
        /// <summary>
        /// execute dotnet run -c Release and choose the benchmarks you want to run
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            KMeansHiggs benchmark = new KMeansHiggs();
            benchmark.TrainKMeansAndLRMaml();
            benchmark.TrainKMeansAndLR();

            MatrixFactorizationCustom custom = new MatrixFactorizationCustom();
            //custom.TrainMatrixFactorizationModel();

            Console.WriteLine("Press any key");
         }

    }
}
