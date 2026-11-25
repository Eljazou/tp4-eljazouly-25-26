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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 4 : pas de RAG quand ce n'est pas utile.
 * Un QueryRouter personnalisé demande au LM si la question porte sur l'IA ;
 * si la réponse est "non", il ne retourne aucun ContentRetriever et il n'y a pas de RAG.
 *
 * @author Abdessamad El Jazouly
 */
public class Test3 {

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

        // ===== Phase 1 : seulement le support de cours sur le RAG =====
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(600, 60).split(document);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 =====
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Template de prompt pour demander au LM si la requête porte sur l'IA.
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{requete}}' porte sur l'IA ? "
                        + "Réponds seulement par 'oui', 'non' ou 'peut-être'.");

        // Classe interne à la méthode main : routage "RAG ou pas RAG".
        // Le LM est interrogé directement avec le ChatModel (pas avec un assistant AiServices).
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                Prompt prompt = promptTemplate.apply(Map.of("requete", query.text()));
                String reponse = model.chat(prompt.text());
                System.out.println("Réponse du LM pour le routage : " + reponse);
                if (reponse.toLowerCase().contains("non")) {
                    // Pas de RAG
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

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
