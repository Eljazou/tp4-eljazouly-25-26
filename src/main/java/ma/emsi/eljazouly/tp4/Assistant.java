package ma.emsi.eljazouly.tp4;

/**
 * Assistant conversationnel utilisé par tous les tests du TP 4.
 * L'implémentation est fournie par LangChain4j (AiServices).
 *
 * @author Abdessamad El Jazouly
 */
public interface Assistant {
    // Prend un message de l'utilisateur et retourne une réponse du LLM.
    String chat(String userMessage);
}
