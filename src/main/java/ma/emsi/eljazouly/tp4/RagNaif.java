package ma.emsi.eljazouly.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 1 : RAG naïf, avec toutes les étapes du RAG explicites (pas de RAG facile).
 * Le PDF du support de cours sur le RAG est dans src/main/resources : Maven copie ce
 * répertoire à la racine du classpath, le fichier peut donc être chargé comme une ressource
 * avec ClassPathDocumentLoader, quel que soit le répertoire de lancement de l'application.
 *
 * @author Abdessamad El Jazouly
 */
public class RagNaif {

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Température basse : le RAG sert à mieux contrôler l'exactitude des réponses.
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash")
                .temperature(0.3)
                // Logging des requêtes envoyées au LLM et de ses réponses
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 du RAG : enregistrement des embeddings =====

        // Création du Document à partir du fichier PDF, avec le parseur Apache Tika.
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // Découpage du document en segments (au plus 600 caractères, 60 de chevauchement).
        DocumentSplitter splitter = DocumentSplitters.recursive(600, 60);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments : " + segments.size());

        // Modèle d'embeddings exécuté localement.
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // embedAll retourne une Response<List<Embedding>> ; content() donne les embeddings.
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // Enregistrement des embeddings et des segments associés dans un magasin en mémoire.
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 du RAG : utilisation des embeddings pour répondre aux questions =====

        // Récupérateur : les 2 segments les plus pertinents, avec un score d'au moins 0,5.
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Assistant avec une mémoire de 10 messages et le ContentRetriever.
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        // Première question, posée directement.
        String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        System.out.println("Question : " + question);
        afficherSegmentsRetrouves(contentRetriever, question);
        System.out.println("Assistant : " + assistant.chat(question));

        // Ensuite, plusieurs questions sans avoir à recompiler.
        conversationAvec(assistant, contentRetriever);
    }

    /**
     * (Optionnel) Affiche les segments retrouvés par le ContentRetriever pour une question.
     */
    private static void afficherSegmentsRetrouves(ContentRetriever contentRetriever, String question) {
        List<Content> contents = contentRetriever.retrieve(Query.from(question));
        System.out.println("--- " + contents.size() + " segment(s) retrouvé(s) pour la question ---");
        for (Content content : contents) {
            System.out.println(content.textSegment().text());
            System.out.println("---");
        }
    }

    private static void conversationAvec(Assistant assistant, ContentRetriever contentRetriever) {
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
                afficherSegmentsRetrouves(contentRetriever, question);
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
