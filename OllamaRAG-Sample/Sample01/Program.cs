#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0003
#pragma warning disable SKEXP0010
#pragma warning disable SKEXP0011
#pragma warning disable SKEXP0050
#pragma warning disable SKEXP0052

using HtmlAgilityPack;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;

var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId: "phi3.5",
    endpoint: new Uri("http://localhost:11434"),
    apiKey: "apikey");
builder.AddLocalTextEmbeddingGeneration();
var kernel = builder.Build();

Console.Clear();
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("Loaded Phi3.5 model.");

// Get the embeddings generator service
var embeddingGenerator = kernel.Services.GetRequiredService<ITextEmbeddingGenerationService>();
var memory = new SemanticTextMemory(new VolatileMemoryStore(), embeddingGenerator);

// Add news to the collection
const string collectionName = "ghc-news";
var allParagraphs = new List<string>();

var articleList = new List<string>
{
    "https://www.linkedin.com/posts/fsoft-ghc_telehealth-telehealth-digitalhealth-activity-7267371560036896768-YIdq",
    "https://www.linkedin.com/posts/fsoft-ghc_techday2024-digitalhealthcare-telehealth-activity-7265257966428184577-b9sl",
    "https://www.linkedin.com/posts/fsoft-ghc_ai-ai-data-activity-7264110087042920449--uqE"
};

var web = new HtmlWeb();
foreach (var article in articleList)
{
    var htmlDoc = web.Load(article);
    var node = htmlDoc.DocumentNode.Descendants(0).FirstOrDefault(n => n.HasClass("attributed-text-segment-list__content"));

    await memory.SaveInformationAsync(collectionName, node.InnerText, Guid.NewGuid().ToString());
}

// Import the text memory plugin into the Kernel.
kernel.ImportPluginFromObject(new TextMemoryPlugin(memory), "memory");
Console.WriteLine($"Import {articleList.Count} article(s) into {collectionName} kernel memory.");
Console.WriteLine();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.Write("Question: ");
    var userInput = Console.ReadLine();
    if (string.IsNullOrEmpty(userInput))
    {
        break;
    }
    Console.ForegroundColor = ConsoleColor.White;

    var prompt = @"
    Question: {{$input}}
    Answer the question using the memory content: {{Recall}}
    If you don't know an answer, say 'I don't know!'";

    var arguments = new KernelArguments
    {
        { "input", userInput },
        { "collection", collectionName }
    };

    var response = kernel.InvokePromptStreamingAsync(prompt, arguments);
    await foreach (var result in response)
    {
        Console.Write(result);
    }
    Console.WriteLine();
    Console.WriteLine();
}

Console.WriteLine();
