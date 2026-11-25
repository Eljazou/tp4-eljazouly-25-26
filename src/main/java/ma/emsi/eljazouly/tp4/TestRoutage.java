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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3 : routage. Deux documents sont enregistrés dans 2 magasins d'embeddings différents :
 * le support de cours sur le RAG et un document sur la cuisine marocaine.
 * Un LanguageModelQueryRouter demande au LM quelle(s) source(s) utiliser pour chaque question.
 *
 * @author Abdessamad El Jazouly
 */
public class TestRoutage {

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
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

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ===== Phase 1 : ingestion des 2 documents dans 2 magasins d'embeddings =====
        EmbeddingStore<TextSegment> embeddingStoreRag = creerEmbeddingStore("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> embeddingStoreCuisine =
                creerEmbeddingStore("cuisine-marocaine.pdf", embeddingModel);

        // ===== Phase 2 : un ContentRetriever par magasin, et un routeur =====
        ContentRetriever contentRetrieverRag = creerContentRetriever(embeddingStoreRag, embeddingModel);
        ContentRetriever contentRetrieverCuisine = creerContentRetriever(embeddingStoreCuisine, embeddingModel);

        // Descriptions en langage naturel des sources, lues par le LM pour choisir la route.
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetrieverRag,
                "Support de cours sur l'intelligence artificielle : le fine-tuning et le RAG "
                        + "(Retrieval Augmented Generation), les embeddings, LangChain4j");
        descriptions.put(contentRetrieverCuisine,
                "Document sur la cuisine marocaine traditionnelle : couscous, tajine, pastilla, "
                        + "harira, thé à la menthe, épices et spécialités régionales");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

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

    /**
     * Phase 1 du RAG pour un fichier PDF de src/main/resources : parsing, découpage en segments,
     * calcul des embeddings et enregistrement dans un magasin d'embeddings en mémoire.
     */
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(String nomFichier,
                                                                  EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(600, 60).split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
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
