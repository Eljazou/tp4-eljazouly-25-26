package ma.emsi.eljazouly.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5 : RAG avec récupération sur le Web.
 * Les informations sont récupérées dans le PDF du cours sur le RAG et sur le Web (Tavily).
 *
 * @author Abdessamad El Jazouly
 */
public class Test4 {

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");
        if (llmKey == null || tavilyKey == null) {
            System.out.println("Les variables d'environnement GEMINI_KEY et TAVILY_KEY doivent être définies.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                // gemini-3.5-flash-lite au lieu de gemini-3.5-flash : quota gratuit journalier de
                // gemini-3.5-flash dépassé (erreur 429 RESOURCE_EXHAUSTED) pendant les tests.
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 : RAG naïf avec le support de cours =====
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(600, 60).split(document);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 =====
        ContentRetriever contentRetrieverPdf = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Moteur de recherche sur le Web Tavily, configuré seulement avec la clé.
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever contentRetrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // Les 2 ContentRetrievers sont toujours utilisés : PDF et Web.
        QueryRouter queryRouter = new DefaultQueryRouter(contentRetrieverPdf, contentRetrieverWeb);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                if (!scanner.hasNextLine()) {
                    break;
                }
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}
